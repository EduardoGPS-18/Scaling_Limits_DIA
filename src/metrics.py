import numpy as np
import torch

from src.utils import image_preprint

def MSE(img_batch, ref_batch, reduction = True):
    if reduction:
        value = torch.nn.MSELoss(size_average=None, reduction='mean')(img_batch, ref_batch)
    else:
        value = torch.nn.MSELoss(size_average=None, reduction='none')(img_batch, ref_batch)
        for _ in range(len(value.shape)-1):
            value = value.mean(dim=-1)
    return value

def image_space_metrics(original, reconstruction, dataset):
    """
    PSNR, SSIM and MSE of a single image after denormalizing both tensors to [0, 1].

    Args:
        original (torch.Tensor): victim image, [C,H,W] or [1,C,H,W], normalized with the dataset statistics.
        reconstruction (torch.Tensor): reconstructed image with the same shape.
        dataset (str): dataset name used to look up the normalization constants.
    """
    o = _to_image(original, dataset)
    r = _to_image(reconstruction, dataset)
    mse = float(np.mean((o - r) ** 2))
    psnr = 100.0 if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))
    return {'psnr': psnr, 'ssim': _global_ssim(o, r), 'mse': mse}


def _to_image(img, dataset):
    """Denormalized float64 array of shape [H,W,C] with values in [0, 1]."""
    img = np.asarray(image_preprint(img.detach().cpu(), scale=dataset), dtype=np.float64)
    img = np.clip(img, 0.0, 1.0)
    return img[..., np.newaxis] if img.ndim == 2 else img


def _global_ssim(a, b, eps=1e-12):
    """SSIM computed over the whole image (no sliding window), averaged over channels."""
    c1, c2 = 1e-4, 9e-4
    values = []
    for c in range(a.shape[-1]):
        x, y = a[..., c], b[..., c]
        mu_x, mu_y = x.mean(), y.mean()
        var_x, var_y = ((x - mu_x) ** 2).mean(), ((y - mu_y) ** 2).mean()
        cov_xy = ((x - mu_x) * (y - mu_y)).mean()
        numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
        values.append(float(numerator / (denominator + eps)))
    return float(np.mean(values))
