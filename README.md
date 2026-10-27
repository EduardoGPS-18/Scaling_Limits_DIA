# Scaling Limits of the Dropout Inversion Attack
This repository contains the experiments of
*Scaling Limits of the Dropout Inversion Attack: An Empirical Analysis Under Increasing Input Dimensionality*
(Eduardo Gordiano Passos Silva and Rodrigo Cesar Pedrosa Silva, UFOP), presented at ENIAC.
IG, DIA and WIIG are run against MLP, LeNet and ResNet models on CIFAR-10 images upscaled from 32x32 to 224x224.

## Based on
This work builds on **Dropout is NOT All You Need to Prevent Gradient Leakage** (Scheliga, Maeder and Seeland, AAAI 2023),
which proposed the Dropout Inversion Attack (DIA), and is a fork of its official implementation:
[github.com/dAI-SY-Group/DropoutInversionAttack](https://github.com/dAI-SY-Group/DropoutInversionAttack).
The paper is available on [ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/26163) and
[arXiv](https://arxiv.org/abs/2208.06163). The DIA, IG and WIIG attacks, the MLP and ViT victims and the
`DIA_demo` notebook come from that repository.

Additions to the original code:
+ `src/models/cnn.py`: LeNet and ResNet18/50 victims with a `MaskDropout` layer before the classification head
+ `freeze_batchnorm` attack option, which keeps BatchNorm layers in eval mode so that the victim gradient is reachable
+ per-sample PSNR, SSIM and MSE in image space (`src/metrics.py`) and per-run exports (`src/reporting.py`)
+ `run_grid.py` / `src/experiment.py`: the experiment grid, with resumable runs and parallel shards
+ `tools/generate_scaled_tdumps.py`: upscaled copies of the CIFAR-10 victim set
+ `Makefile`: shortcuts for installing, generating the data and running the grid

`data/CIFAR10_128.tdump` holds the first 128 images of the CIFAR-10 training set, normalized with the CIFAR-10 mean and std.

## Running the experiments
Requires Python 3.12 (versions used in the paper are pinned in `requirements.txt`, tested with CUDA 12.8).
```
make install                  # .venv with requirements.txt
make smoke                    # smoke test: 150 iterations per attack, written to outputs/debug
make grid SHARDS=3            # full grid: 3 datasets x 3 models x 3 dropout rates x 10 runs
make status                   # completed runs per combination
```
`make help` lists the targets and variables, e.g. `make grid DATASETS=CIFAR10_128_224x224 MODELS="ResNet50 ViT" DROPOUTS=0.5`.
Without make, the same steps are:
```
pip install -r requirements.txt
python tools/generate_scaled_tdumps.py                  # data/CIFAR10_128_{64x64,128x128,224x224}.tdump
python run_grid.py --debug --runs 1 --output-dir outputs/debug
python run_grid.py --shards 3
python run_grid.py --status
```
Every run is written to `outputs/<dataset>_<model>_p<dropout>_seed<seed>_run<i>/`, with one row per attack and
victim image in `metrics.csv`, plus the raw tensors, loss curves and a comparison figure. Finished runs are skipped,
so the same command resumes an interrupted grid. Debug runs use the same directory names as real runs, so keep them
in a separate `--output-dir`: otherwise the full grid skips them as finished. `python run_grid.py --help` lists all options.

## Example reconstructions from the original work
Reconstructions on a ViT [2] for different attacks, dropout rates and datasets, from the original DIA repository.

| | Original| IG [1]                                      | IG [1]                                     | DIA                                       |
|:-:|:------:|:---------------------------------------:|:---------------------------------------:|:-----------------------------------------:|
|Dropout Rate p| - | p=0.00                                  | p=0.10                                  | p=0.10                                    |
|MNIST   | ![mnist_original](demos/MNIST_original.png) | ![mnist_ig_p00](demos/MNIST_IG_p00.gif) | ![mnist_ig_p01](demos/MNIST_IG_p01.gif) | ![mnist_dia_p01](demos/MNIST_DIA_p01.gif) |
|CIFAR-10| ![cifar10_original](demos/CIF_original.png) | ![cifar10_ig_p00](demos/CIF_IG_p00.gif) | ![cifar10_ig_p01](demos/CIF_IG_p01.gif) | ![cifar10_dia_p01](demos/CIF_DIA_p01.gif) |
|ImageNet| ![ImageNet_original](demos/ImageNet_original.png) | ![ImageNet_ig_p00](demos/ImageNet_IG_p00.gif) | ![ImageNet_ig_p01](demos/ImageNet_IG_p01.gif) | ![ImageNet_dia_p01](demos/ImageNet_DIA_p01.gif) |

## Credits
+ The Dropout Inversion Attack and the code this repository extends:
  [DropoutInversionAttack](https://github.com/dAI-SY-Group/DropoutInversionAttack) by Daniel Scheliga, Patrick Mäder and Marco Seeland
+ [1] [invertinggradients](https://github.com/JonasGeiping/invertinggradients), the implementation of [IG](https://arxiv.org/abs/2003.14053) the original code is based on
+ [2] [vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py), the implementation of the VisionTransformer

## Acknowledgements
We thank Daniel Scheliga, Patrick Mäder and Marco Seeland for publishing the implementation of the Dropout Inversion
Attack, without which this work would not have been possible, and the authors of the repositories listed above
for making their code available.

If you use this code, please also cite the original work:
```
@article{scheliga2023dropout,
  title={Dropout Is NOT All You Need to Prevent Gradient Leakage},
  volume={37},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/26163},
  DOI={10.1609/aaai.v37i8.26163},
  number={8},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Scheliga, Daniel and Maeder, Patrick and Seeland, Marco},
  year={2023},
  month={Jun.},
  pages={9733-9741}
}
```

<details>
<summary>Abstract of the original work</summary>

Gradient inversion attacks on federated learning systems reconstruct client training data from exchanged gradient information. To defend against such attacks, a variety of defense mechanisms were proposed. However, they usually lead to an unacceptable trade-off between privacy and model utility. Recent observations suggest that dropout could mitigate gradient leakage and improve model utility if added to neural networks. Unfortunately, this phenomenon has not been systematically researched yet. In this work, we thoroughly analyze the effect of dropout on iterative gradient inversion attacks. We find that state of the art attacks are not able to reconstruct the client data due to the stochasticity induced by dropout during model training. Nonetheless, we argue that dropout does not offer reliable protection if the dropout induced stochasticity is adequately modeled during attack optimization. Consequently, we propose a novel Dropout Inversion Attack (DIA) that jointly optimizes for client data and dropout masks to approximate the stochastic client model. We conduct an extensive systematic evaluation of our attack on four seminal model architectures and three image classification datasets of increasing complexity. We find that our proposed attack bypasses the protection seemingly induced by dropout and reconstructs client data with high fidelity. Our work demonstrates that privacy inducing changes to model architectures alone cannot be assumed to reliably protect from gradient leakage and therefore should be combined with complementary defense mechanisms.
</details>

## Requirements
+ matplotlib
+ munch
+ pyyaml
+ numpy
+ pytorch
+ einops
+ torchvision (LeNet/ResNet victims)

The original conda environment of the DIA repository (Python 3.9) is kept in `environment.yaml`:
```
conda env create -f environment.yaml
conda activate DIA
```
