"""
Generate upscaled copies of the CIFAR-10 victim set for the input-resolution experiments.

    python tools/generate_scaled_tdumps.py                # 64x64, 128x128 and 224x224
    python tools/generate_scaled_tdumps.py --sizes 512

Reads data/CIFAR10_128.tdump (128 normalized 3x32x32 images) and writes
data/CIFAR10_128_{size}x{size}.tdump for each size. Images are resized with bilinear
interpolation and keep the CIFAR-10 normalization (see src/constants.yaml).
Existing files are left untouched.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
SOURCE = DATA_DIR / 'CIFAR10_128.tdump'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sizes', nargs='+', type=int, default=[64, 128, 224])
    args = parser.parse_args()

    victims = torch.load(SOURCE)
    print(f'Loaded {SOURCE.name} with inputs of shape {tuple(victims["inputs"].shape)}')
    for size in args.sizes:
        out_path = DATA_DIR / f'CIFAR10_128_{size}x{size}.tdump'
        if out_path.exists():
            print(f'{out_path.name} already exists, skipping')
            continue
        inputs = F.interpolate(victims['inputs'], size=(size, size), mode='bilinear', align_corners=False)
        torch.save({'inputs': inputs, 'targets': victims['targets']}, out_path)
        print(f'Saved {out_path.name} with inputs of shape {tuple(inputs.shape)}')


if __name__ == '__main__':
    main()
