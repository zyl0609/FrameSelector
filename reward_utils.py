import torch
import torch.nn.functional as F

def gaussian_window(window_size: int, sigma: float, channels: int = 3):
    """
    生成 (window_size, window_size) 的 2D 高斯核，重复 channels 份
    返回 shape: (channels, 1, window_size, window_size)
    """
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    g = g.outer(g)                      # (w,w)
    g = g.view(1, 1, window_size, window_size)
    return g.expand(channels, 1, -1, -1).contiguous()


def ssim_map(x: torch.Tensor, y: torch.Tensor,
             window_size: int = 11, sigma: float = 1.5,
             C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
    """
    Compute pixel-wise SSIM map.

    :param x:
    :param y:
    :param window_size:
    :param sigma:
    :param C1:
    :param C2:

    :return ssim:(B, 3, H, W)
    """
    B, C, H, W = x.shape
    assert x.shape == y.shape
    window = gaussian_window(window_size, sigma, C).to(device=x.device, dtype=x.dtype)

    # 均值 μx, μy
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=C)

    # 方差 σx², σy², 协方差 σxy
    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=C) - mu_x ** 2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=C) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=C) - mu_x * mu_y

    # SSIM 分子 & 分母
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim = numerator / (denominator + 1e-8)
    return ssim                                    # (B, C, H, W) 逐像素 SSIM


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, reduce: str = 'mean'):
    """
    1 - SSIM 作为 loss，返回标量
    reduce='mean' 或 'none'（返回 map）
    """
    ssim = ssim_map(x, y, window_size)             # (B, C, H, W)
    if reduce == 'mean':
        return 1 - ssim.mean()
    return 1 - ssim                                # 需要 map 时拿去做 per-pixel weight