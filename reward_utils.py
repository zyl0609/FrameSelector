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
    计算 SSIM 逐像素 map，返回 shape (B, 3, H, W)
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



if __name__ == "__main__":
    # ---------- 1. 构造假数据 ----------
    B, C, H, W = 4, 3, 128, 128
    drop_render = torch.rand(B, C, H, W, requires_grad=True)   # 预测图
    gt_render   = torch.rand(B, C, H, W)                       # 伪 GT

    # ---------- 2. 计算联合 reward ----------
    alpha = 0.5
    l1   = F.l1_loss(drop_render, gt_render)
    ssim = ssim_loss(drop_render, gt_render, window_size=11)
    reward = -(alpha * l1 + (1 - alpha) * ssim)

    print(f"L1        = {l1.item():.4f}")
    print(f"SSIM-loss = {ssim.item():.4f}")
    print(f"reward    = {reward.item():.4f}")   # 期望 ≈ -0.25 左右

    # ---------- 3. 梯度检查 ----------
    reward.backward()
    print(f"drop_render.grad norm = {drop_render.grad.norm().item():.4f}")

    # ---------- 4. 模拟 keep-ratio ----------
    # 假装 selector 输出 mask，随机生成
    mask = torch.rand(B, 1, 1, 1)           # 0~1
    keep_ratio = mask.mean().item()
    print(f"mock keep_ratio = {keep_ratio:.2%}")

    # ---------- 5. 组装完整训练 step ----------
    sparse_coeff = 0.15
    sparse = 1.0 - keep_ratio
    total_reward = reward + sparse_coeff * sparse
    print(f"sparse term = {sparse_coeff * sparse:.4f}")
    print(f"total reward= {total_reward.item():.4f}")