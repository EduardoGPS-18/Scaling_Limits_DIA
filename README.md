# Dropout is NOT All You Need to Prevent Gradient Leakage
This repository contains the implementation of our proposed Dropout Inversion Attack (DIA).
For a demonstration of our attack follow the DIA_demo jupyter notebook.

The paper including all empirical results can be found on [ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/26163) and [arXiv](https://arxiv.org/abs/2208.06163)

## Scaling Limits of the Dropout Inversion Attack
This fork extends the original implementation with the experiments of
*Scaling Limits of the Dropout Inversion Attack: An Empirical Analysis Under Increasing Input Dimensionality*
(Eduardo Gordiano Passos Silva and Rodrigo Cesar Pedrosa Silva, UFOP), presented at ENIAC.
IG, DIA and WIIG are run against MLP, LeNet and ResNet models on CIFAR-10 images upscaled from 32x32 to 224x224.

Additions to the original code:
+ `src/models/cnn.py`: LeNet and ResNet18/50 victims with a `MaskDropout` layer before the classification head
+ `freeze_batchnorm` attack option, which keeps BatchNorm layers in eval mode so that the victim gradient is reachable
+ per-sample PSNR, SSIM and MSE in image space (`src/metrics.py`) and per-run exports (`src/reporting.py`)
+ `run_grid.py` / `src/experiment.py`: the experiment grid, with resumable runs and parallel shards
+ `tools/generate_scaled_tdumps.py`: upscaled copies of the CIFAR-10 victim set

`data/CIFAR10_128.tdump` holds the first 128 images of the CIFAR-10 training set, normalized with the CIFAR-10 mean and std.

### Running the experiments
```
pip install -r requirements.txt
python tools/generate_scaled_tdumps.py          # data/CIFAR10_128_{64x64,128x128,224x224}.tdump
python run_grid.py --debug --runs 1             # smoke test: 150 iterations per attack
python run_grid.py --shards 3                   # full grid: 3 datasets x 3 models x 3 dropout rates x 10 runs
python run_grid.py --status                     # completed runs per combination
```
Every run is written to `outputs/<dataset>_<model>_p<dropout>_seed<seed>_run<i>/`, with one row per attack and
victim image in `metrics.csv`, plus the raw tensors, loss curves and a comparison figure. Finished runs are skipped,
so the same command resumes an interrupted grid. `python run_grid.py --help` lists all options, e.g.
`--datasets CIFAR10_128_224x224 --models ResNet50 ViT --dropouts 0.5`.

## Example reconstructions for different attacks, dropout rates and datasets on a ViT [2]
| | Original| IG [1]                                      | IG [1]                                     | DIA (Ours)                                       |
|:-:|:------:|:---------------------------------------:|:---------------------------------------:|:-----------------------------------------:|
|Dropout Rate p| - | p=0.00                                  | p=0.10                                  | p=0.10                                    |
|MNIST   | ![mnist_original](demos/MNIST_original.png) | ![mnist_ig_p00](demos/MNIST_IG_p00.gif) | ![mnist_ig_p01](demos/MNIST_IG_p01.gif) | ![mnist_dia_p01](demos/MNIST_DIA_p01.gif) |
|CIFAR-10| ![cifar10_original](demos/CIF_original.png) | ![cifar10_ig_p00](demos/CIF_IG_p00.gif) | ![cifar10_ig_p01](demos/CIF_IG_p01.gif) | ![cifar10_dia_p01](demos/CIF_DIA_p01.gif) |
|ImageNet| ![ImageNet_original](demos/ImageNet_original.png) | ![ImageNet_ig_p00](demos/ImageNet_IG_p00.gif) | ![ImageNet_ig_p01](demos/ImageNet_IG_p01.gif) | ![ImageNet_dia_p01](demos/ImageNet_DIA_p01.gif) |

## Please cite as:
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

## Abstract:
Gradient inversion attacks on federated learning systems reconstruct client training data from exchanged gradient information. To defend against such attacks, a variety of defense mechanisms were proposed. However, they usually lead to an unacceptable trade-off between privacy and model utility. Recent observations suggest that dropout could mitigate gradient leakage and improve model utility if added to neural networks. Unfortunately, this phenomenon has not been systematically researched yet. In this work, we thoroughly analyze the effect of dropout on iterative gradient inversion attacks. We find that state of the art attacks are not able to reconstruct the client data due to the stochasticity induced by dropout during model training. Nonetheless, we argue that dropout does not offer reliable protection if the dropout induced stochasticity is adequately modeled during attack optimization. Consequently, we propose a novel Dropout Inversion Attack (DIA) that jointly optimizes for client data and dropout masks to approximate the stochastic client model. We conduct an extensive systematic evaluation of our attack on four seminal model architectures and three image classification datasets of increasing complexity. We find that our proposed attack bypasses the protection seemingly induced by dropout and reconstructs client data with high fidelity. Our work demonstrates that privacy inducing changes to model architectures alone cannot be assumed to reliably protect from gradient leakage and therefore should be combined with complementary defense mechanisms.

## Requirements:
+ matplotlib
+ munch
+ pyyaml
+ numpy
+ pytorch
+ einops
+ torchvision (LeNet/ResNet victims)

You can also use [conda](https://www.anaconda.com/) to recreate our virtual environment:
```
conda env create -f environment.yaml
conda activate DIA
```

## Credits:
We base our implementation on the following repositories:
+ [1] [GitHub](https://github.com/JonasGeiping/invertinggradients) for the implementation of [IG](https://arxiv.org/abs/2003.14053)
+ [2] [GitHub](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py) for the implementation of the VisionTransformer
