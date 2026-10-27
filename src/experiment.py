"""
Experiment grid: IG, DIA and WIIG against every (dataset, model, dropout rate) combination.

Each combination is repeated ``runs`` times. Run ``i`` uses seed ``seed + i - 1``, which picks
the victim batches, initializes the model and seeds the victim's dropout masks, so runs are
independent blocks while the three attacks inside a run stay paired (same model, same victims).

Resilient to interrupted runs:
- Every run is exported to its own directory
- Skipped when that directory already holds a ``metrics.csv``
- Resumes where it stopped
- Parallel shards never collide
"""

import gc
import random
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import torch

from src import attacks
from src.data import load_victim_data
from src.models import get_model
from src.reporting import export_run
from src.utils import build_config, get_device, set_seeds

CIFAR_10_NUM_CLASSES = 10
ROOT = Path(__file__).resolve().parent.parent

# attack name -> (config file in example_configs/attacks, attack class)
ATTACKS = {
    'IG': ('IG.yaml', attacks.IG),
    'DIA': ('DIA.yaml', attacks.DropoutInversionAttack),
    'WIIG': ('WIIG.yaml', attacks.IG),  # IG with access to the victim's dropout masks (imitate_dropout)
}
DEBUG_MAX_ITERATIONS = 150


@dataclass
class GridConfig:
    datasets: list = field(default_factory=lambda: ['CIFAR10_128', 'CIFAR10_128_64x64', 'CIFAR10_128_128x128'])
    models: list = field(default_factory=lambda: ['MLP', 'LeNet', 'ResNet18'])
    dropout_rates: list = field(default_factory=lambda: [0.0, 0.25, 0.5])
    runs: int = 10
    seed: int = 42
    victim_batchsize: int = 2
    num_batches: int = 1
    debug: bool = False
    output_dir: Path = ROOT / 'outputs'


@dataclass(frozen=True)
class Combination:
    dataset: str
    model: str
    dropout_rate: float

    def __str__(self):
        return f'{self.dataset} | {self.model} | p={self.dropout_rate}'


def combinations(config, shard_index=0, shard_count=1):
    """
    Grid combinations, ordered by dataset first.

    - Datasets are listed in increasing resolution, so the cheap combinations finish first.
    - Shards take a round-robin slice, which keeps their workloads balanced.
    """
    grid = [
        Combination(dataset, model, float(dropout))
        for dataset, model, dropout
        in product(config.datasets, config.models, config.dropout_rates)
    ]
    return grid[shard_index::shard_count]

def run_dir(config, combination, run):
    tag = f'{combination.dataset}_{combination.model}_p{combination.dropout_rate}_seed{config.seed}_run{run}'
    return Path(config.output_dir) / tag


def is_done(path):
    return (Path(path) / 'metrics.csv').exists()


def run_grid(config, shard_index=0, shard_count=1):
    """Run (or resume) the grid slice that belongs to this shard."""
    grid = combinations(config, shard_index, shard_count)
    total = len(grid) * config.runs
    print(f'shard {shard_index + 1}/{shard_count}: {len(grid)} combinations x {config.runs} runs = {total} runs', flush=True)

    device = get_device()
    victim_data = {}

    combinations_runs = product[tuple](grid, range(1, config.runs + 1))
    for n, (combination, run) in enumerate[tuple[Combination, int]](combinations_runs, start=1):
        out_dir = run_dir(config, combination, run)
        if is_done(out_dir):
            print(f'[{n}/{total}] {out_dir.name}: already done, skipping', flush=True)
            continue
        
        print(f'[{n}/{total}] {out_dir.name}: running', flush=True)
        if combination.dataset not in victim_data:
            victim_data[combination.dataset] = load_victim_data(
                str(ROOT / 'data' / f'{combination.dataset}.tdump'),
                config.victim_batchsize
            )
        
        started = time.perf_counter()
        run_once(config, combination, run, victim_data[combination.dataset], device, out_dir)
        wall_clock_s = time.perf_counter() - started

        print(f'[{n}/{total}] {out_dir.name}: saved ({wall_clock_s:.0f} s)', flush=True)


def run_once(config, combination, run, victim_data, device, out_dir):
    """Attack one freshly initialized model with IG, DIA and WIIG and export the results."""
    run_seed = config.seed + run - 1
    victim_inputs, victim_targets, data_shape = victim_data
    num_classes = CIFAR_10_NUM_CLASSES

    set_seeds(run_seed)
    selection = random.sample(range(len(victim_targets)), config.num_batches)
    inputs = [victim_inputs[i] for i in selection]
    targets = [victim_targets[i] for i in selection]

    model_config = build_config(str(ROOT / 'example_configs' / 'models' / f'{combination.model}.yaml'))
    model_config.dropout_rate = combination.dropout_rate
    model_config.data_shape = data_shape
    model_config.num_classes = num_classes
    
    set_seeds(run_seed)
    model = get_model(model_config.name, model_config).to(device)

    packages = {}
    for name, (config_file, attack_class) in ATTACKS.items():
        print(f'{name} setup:', flush=True)
        attack_config = build_config(str(ROOT / 'example_configs' / 'attacks' / config_file))
        attack_config.device = device
        attack_config.num_classes = num_classes
        attack_config.victim_batchsize = config.victim_batchsize
        if config.debug:
            attack_config.max_iterations = DEBUG_MAX_ITERATIONS
        attacker = attack_class(combination.dataset, model, inputs, targets, attack_config)
        # Reseeding before every attack makes the victim's dropout masks identical across the three attacks.
        set_seeds(run_seed)

        attack_started = time.perf_counter()
        reconstructions, results = attacker.attack()
        packages[name] = {
            'reconstructions': reconstructions,
            'results': results,
            'wall_clock_s': time.perf_counter() - attack_started,
        }

    export_run(out_dir, combination.dataset, combination.model, combination.dropout_rate, run_seed, packages)

    del model, packages
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
