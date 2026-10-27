"""
Persist the artifacts of one experiment run.

Layout of a run directory::

    meta.json                    run description (dataset, model, dropout rate, seed, attacks)
    metrics.csv                  one row per (attack, batch, sample); written last, marks the run as done
    tensors.pt                   victim inputs, targets and reconstructions, to recompute any metric later
    comparison_batch{b}.png      original next to every attack's reconstruction
    loss/{attack}_batch{b}.csv   reconstruction loss per iteration
    loss/comparison_batch{b}.png loss curves of all attacks
"""
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.metrics import image_space_metrics
from src.utils import image_preprint

METRICS_FIELDS = [
    'run_tag', 'dataset', 'model', 'dropout_rate', 'seed', 'attack',
    'batch', 'sample', 'label', 'channels', 'height', 'width',
    'psnr', 'ssim', 'mse', 'recon_loss_best', 'best_iteration',
    'needed_iterations', 'iters_ran', 'wall_clock_s',
]


def export_run(out_dir, dataset, model_name, dropout_rate, seed, packages):
    """
    Write all artifacts of a run to ``out_dir``.

    Args:
        out_dir (Path): run directory, created if missing. Its name is used as the run tag.
        dataset (str): dataset name, also used to denormalize the images.
        model_name (str): name of the attacked architecture.
        dropout_rate (float): dropout rate of the attacked model.
        seed (int): seed of this run.
        packages (dict): attack name -> {'reconstructions', 'results', 'wall_clock_s'}, where
            reconstructions and results are the two lists returned by ``attack()``.
    """
    out_dir = Path(out_dir)
    (out_dir / 'loss').mkdir(parents=True, exist_ok=True)
    reference = next(iter(packages.values()))['reconstructions']
    data_shape = list(reference[0]['Inputs'].shape[-3:])

    meta = {
        'dataset': dataset,
        'model': model_name,
        'dropout_rate': dropout_rate,
        'seed': seed,
        'run_tag': out_dir.name,
        'attacks': list(packages),
        'data_shape': data_shape,
        'n_batches': len(reference),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    rows = []
    tensors = {}
    for b, batch in enumerate(reference):
        inputs, targets = batch['Inputs'], batch['Targets']
        reconstructions = {name: pack['reconstructions'][b]['Reconstructions'] for name, pack in packages.items()}
        results = {name: pack['results'][b] for name, pack in packages.items()}

        for name, pack in packages.items():
            result = results[name]
            for s in range(inputs.shape[0]):
                rows.append({
                    'run_tag': out_dir.name,
                    'dataset': dataset,
                    'model': model_name,
                    'dropout_rate': dropout_rate,
                    'seed': seed,
                    'attack': name,
                    'batch': b,
                    'sample': s,
                    'label': int(targets[s]),
                    'channels': data_shape[0],
                    'height': data_shape[1],
                    'width': data_shape[2],
                    **image_space_metrics(inputs[s], reconstructions[name][s], dataset),
                    'recon_loss_best': _value_at(result.get('ReconstructionLoss'), s),
                    'best_iteration': _value_at(result.get('BestIteration'), s),
                    'needed_iterations': _value_at(result.get('NeededIterations'), s),
                    'iters_ran': len(result.get('LossHistory') or []),
                    'wall_clock_s': pack['wall_clock_s'],
                })

        loss_histories = {name: result.get('LossHistory') or [] for name, result in results.items()}
        for name, history in loss_histories.items():
            if history:
                np.savetxt(out_dir / 'loss' / f'{name}_batch{b}.csv', np.asarray(history), delimiter=',',
                           header='reconstruction_loss', comments='')
        if any(loss_histories.values()):
            _plot_losses(loss_histories, out_dir / 'loss' / f'comparison_batch{b}.png', title=f'Batch {b}')
        _plot_reconstructions(inputs, reconstructions, dataset, out_dir / f'comparison_batch{b}.png')

        tensors[f'batch{b}'] = {
            'inputs': inputs.detach().cpu(),
            'targets': targets.detach().cpu(),
            'reconstructions': {name: rec.detach().cpu() for name, rec in reconstructions.items()},
        }

    # metrics.csv is the marker of a finished run, so it is written last and atomically: an
    # interrupted run never looks finished while its tensors are missing or half written.
    _atomic_write(out_dir / 'tensors.pt', lambda path: torch.save(tensors, path))
    _atomic_write(out_dir / 'metrics.csv', lambda path: _write_csv(rows, path))
    return out_dir


def _value_at(metric, sample):
    """Attack metrics are a scalar for batchsize 1 and one value per sample otherwise."""
    if metric is None:
        return float('nan')
    values = torch.as_tensor(metric).detach().cpu().reshape(-1)
    return float(values[0] if values.numel() == 1 else values[sample])


def _atomic_write(path, write):
    tmp = path.with_name(path.name + '.tmp')
    write(tmp)
    os.replace(tmp, path)


def _write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def _plot_reconstructions(inputs, reconstructions, dataset, path):
    """One row per sample: original | one column per attack."""
    n_samples, n_cols = inputs.shape[0], 1 + len(reconstructions)
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(3 * n_cols, 3 * n_samples), dpi=120, squeeze=False)
    for s in range(n_samples):
        columns = [('original', inputs[s])] + [(name, rec[s]) for name, rec in reconstructions.items()]
        for ax, (title, img) in zip(axes[s], columns):
            ax.imshow(image_preprint(img, scale=dataset))
            ax.set_title(title if s == 0 else '')
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def _plot_losses(loss_histories, path, title):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    for name, history in loss_histories.items():
        if history:
            ax.plot(np.arange(1, len(history) + 1), history, linewidth=1.0, label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reconstruction loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
