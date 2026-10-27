"""
Run the experiment grid (IG + DIA + WIIG) from the command line.

    python run_grid.py                               # default grid, single process
    python run_grid.py --shards 3                    # three worker processes sharing the GPU
    python run_grid.py --debug --runs 1              # 150 iterations per attack, to validate the pipeline
    python run_grid.py --datasets CIFAR10_128_224x224 --models ResNet50 --dropouts 0.5
    python run_grid.py --status                      # completed runs per combination

Finished runs are skipped, so the same command resumes an interrupted grid. With ``--shards``
each worker writes its output to logs/shard{i}.log while this process reports overall progress.
"""
import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # headless: figures are only written to disk

from src.experiment import ROOT, GridConfig, combinations, is_done, run_dir, run_grid  # noqa: E402

LOG_DIR = ROOT / 'logs'


def parse_args():
    defaults = GridConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=defaults.datasets,
                        help='victim sets in data/ (file name without .tdump)')
    parser.add_argument('--models', nargs='+', default=defaults.models,
                        help='model configs in example_configs/models (file name without .yaml)')
    parser.add_argument('--dropouts', nargs='+', type=float, default=defaults.dropout_rates)
    parser.add_argument('--runs', type=int, default=defaults.runs, help='independent runs per combination')
    parser.add_argument('--seed', type=int, default=defaults.seed, help='seed of the first run')
    parser.add_argument('--output-dir', type=Path, default=defaults.output_dir)
    parser.add_argument('--debug', action='store_true',
                        help='cap every attack at 150 iterations to validate the pipeline')
    parser.add_argument('--shards', type=int, default=1, help='number of parallel worker processes')
    parser.add_argument('--shard-index', type=int, help=argparse.SUPPRESS)  # set for worker processes
    parser.add_argument('--status', action='store_true', help='show completed runs per combination and exit')
    return parser.parse_args()


def main():
    args = parse_args()
    config = GridConfig(datasets=args.datasets, models=args.models, dropout_rates=args.dropouts, runs=args.runs,
                        seed=args.seed, debug=args.debug, output_dir=args.output_dir)
    if args.status:
        return print_status(config)
    if args.shard_index is not None:
        return run_grid(config, args.shard_index, args.shards)
    if args.shards == 1:
        return run_grid(config)
    return run_shards(config, args.shards)


def print_status(config):
    grid = combinations(config)
    finished = 0
    print(f'{"dataset":<24} {"model":<10} {"dropout":>8} {"runs":>8}')
    for combination in grid:
        done = sum(is_done(run_dir(config, combination, run)) for run in range(1, config.runs + 1))
        finished += done
        print(f'{combination.dataset:<24} {combination.model:<10} {combination.dropout_rate:>8} '
              f'{done:>4}/{config.runs}')
    print(f'\n{finished}/{len(grid) * config.runs} runs finished in {config.output_dir}')
    return 0


def run_shards(config, shard_count, refresh_s=10.0):
    """Start one worker process per shard and report progress until all of them exit."""
    expected = [run_dir(config, c, r) for c in combinations(config) for r in range(1, config.runs + 1)]
    already_done = sum(is_done(path) for path in expected)
    LOG_DIR.mkdir(exist_ok=True)

    workers = []
    for index in range(shard_count):
        log_path = LOG_DIR / f'shard{index}.log'
        with open(log_path, 'w') as log:
            process = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], '--shard-index', str(index)],
                                       stdout=log, stderr=subprocess.STDOUT)
        workers.append((process, log_path))
        print(f'shard {index} started (pid {process.pid}), log: {log_path.relative_to(ROOT)}')

    # Treat SIGTERM like Ctrl+C so that killing this process also stops its workers.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    started = time.perf_counter()
    try:
        while any(process.poll() is None for process, _ in workers):
            time.sleep(refresh_s)
            print_progress(expected, already_done, started)
    except KeyboardInterrupt:
        print('\nstopping workers; finished runs are kept and the same command resumes the grid')
        for process, _ in workers:
            process.terminate()
        for process, _ in workers:
            process.wait()
        return 130
    print_progress(expected, already_done, started)
    print()

    failed = [(process, log_path) for process, log_path in workers if process.returncode != 0]
    for process, log_path in failed:
        print(f'\nshard failed with exit code {process.returncode}, end of {log_path.relative_to(ROOT)}:')
        print('\n'.join(log_path.read_text(errors='replace').splitlines()[-15:]))
    return 1 if failed else 0


def print_progress(expected, already_done, started):
    done = sum(is_done(path) for path in expected)
    elapsed = time.perf_counter() - started
    new_runs = done - already_done
    eta = f'{(len(expected) - done) * elapsed / new_runs / 3600:.1f} h left' if new_runs else 'estimating...'
    print(f'\r{done}/{len(expected)} runs finished | {elapsed / 3600:.1f} h elapsed | {eta}   ', end='', flush=True)


if __name__ == '__main__':
    sys.exit(main())
