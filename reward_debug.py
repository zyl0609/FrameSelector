import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 你的清晰度函数（已修正）
def clarity_reward_from_image(
    rgb_image: torch.Tensor
) -> torch.Tensor:
    """
    Compute clarity reward from RGB images based on spatial gradients.

    :param rgb_image: Tensor of shape (1, S, 3, H, W) or (S, 3, H, W) representing RGB images.

    :return: Tensor of shape (S,) representing clarity scores for each image.
    """
    # process batch dimension if exists
    if rgb_image.dim() == 5:
        rgb_image = rgb_image.squeeze(0)  # (S, 3, H, W)

    S, C, H, W = rgb_image.shape
    device = rgb_image.device

    gray = rgb_image.mean(dim=1, keepdim=True)  # (S, 1, H, W)

    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    
    # gradient (S, 1, H, W)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)

    # average over spatial dimensions
    clarity = gradient_magnitude.mean(dim=[-2, -1]).squeeze()

    # normalization (Robust method)
    # use 95th percentile to avoid extreme values
    percentile_95 = torch.quantile(clarity, 0.95)
    clarity = clarity / (percentile_95 + 1e-8)
    clarity = torch.clamp(clarity, 0.0, 1.0)

    return clarity  # (S,)


def clarity_reward_from_image_fft(
    rgb_image: torch.Tensor, 
    low_freq_radius_ratio: float = 0.08,
    patch_size: int = 32,
    pad_mode: str = 'reflect'  # 'reflect' or 'constant'
) -> torch.Tensor:
    """
    Compute clarity reward from RGB images based on high-frequency energy in the frequency domain.

    :param rgb_image: Tensor of shape (1, S, 3, H, W) or (S, 3, H, W) representing RGB images.
    :param low_freq_radius_ratio: Ratio of low-frequency radius to image size to be filtered out.
    :param patch_size: Size to which images are resized for FFT computation.

    :return: Tensor of shape (S,) representing clarity scores for each image.
    """
   # 统一维度 (S, 3, H, W)
    if rgb_image.dim() == 5:
        rgb_image = rgb_image.squeeze(0)
    S, C, H, W = rgb_image.shape
    
    # convert to grayscale
    gray = rgb_image.mean(dim=1, keepdim=True)

    # compute padding sizes
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    # symmetric padding
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # padding
    gray_padded = F.pad(gray, (pad_left, pad_right, pad_top, pad_bottom), 
                        mode=pad_mode, value=0)
    
    _, _, H_pad, W_pad = gray_padded.shape

    # (S, 1, n_patches_h, n_patches_w, patch_size, patch_size)
    n_patches_h = H_pad // patch_size
    n_patches_w = W_pad // patch_size
    patches = gray_padded.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # (S*n_patches_h*n_patches_w, patch_size, patch_size)
    patches_flat = patches.contiguous().view(-1, patch_size, patch_size)
    
    # FFT transform and move to center
    freq = torch.fft.fft2(patches_flat)
    freq_shift = torch.fft.fftshift(freq, dim=(-2, -1))
    
    magnitude = torch.abs(freq_shift)
    
    # 构建高频掩码
    center = patch_size // 2
    Y, X = torch.meshgrid(
        torch.arange(patch_size, device=gray.device), 
        torch.arange(patch_size, device=gray.device), 
        indexing='ij'
    )
    radius = torch.sqrt((X - center)**2 + (Y - center)**2).float()
    mask = (radius > center * low_freq_radius_ratio).float()
    
    # 计算每块的高频能量占比
    high_freq_energy = (magnitude * mask).sum(dim=(-1, -2))
    total_energy = magnitude.sum(dim=(-1, -2)) + 1e-8
    high_freq_ratio = high_freq_energy / total_energy
    
    # 重塑回每帧的patch矩阵 (S, n_patches_h, n_patches_w)
    high_freq_ratio = high_freq_ratio.view(S, n_patches_h, n_patches_w)
    
    # 空间平均：每帧的整体清晰度 (S,)
    clarity_per_frame = high_freq_ratio.mean(dim=(-1, -2))
    
    # 序列内归一化
    percentile_95 = torch.quantile(clarity_per_frame, 0.95)
    clarity_per_frame = clarity_per_frame / (percentile_95 + 1e-8)
    clarity_per_frame = torch.clamp(clarity_per_frame, 0.0, 1.0)
    
    return clarity_per_frame

def load_frames_from_sequence(sequence_dir: Path, num_frames: int = 100):
    """从7Scenes序列加载帧"""
    # 获取所有png文件
    frame_paths = sorted(sequence_dir.glob("frame-*.color.png"))[:num_frames]
    
    frames = []
    for fp in frame_paths:
        # 打开并转换为tensor (0~1)
        img = Image.open(fp).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
        frames.append(img_tensor)
    
    return torch.stack(frames)  # (num_frames, 3, H, W)

def visualize_clarity_results(frames, clarity_scores, method_name=""):
    """可视化最清晰/最模糊的帧"""
    # 找出最清晰和最模糊的5帧
    _, top_indices = torch.topk(clarity_scores, 5)
    _, bottom_indices = torch.topk(clarity_scores, 5, largest=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Clarity Scores - {method_name}", fontsize=16)
    
    # 显示最清晰的帧
    for i, idx in enumerate(top_indices):
        ax = axes[0, i]
        img = frames[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Score: {clarity_scores[idx]:.3f}\nFrame: {idx.item()}")
        ax.axis('off')
    axes[0, 0].set_ylabel("Most Clear", fontsize=12)
    
    # 显示最模糊的帧
    for i, idx in enumerate(bottom_indices):
        ax = axes[1, i]
        img = frames[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Score: {clarity_scores[idx]:.3f}\nFrame: {idx.item()}")
        ax.axis('off')
    axes[1, 0].set_ylabel("Most Blurry", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"clarity_comparison_{method_name}.png", dpi=150)
    plt.show()

# 主测试流程
def test_clarity_on_real_data():
    # 路径设置
    seq_dir = Path("/home/lidabao/projects/code/cvpr2026/data/7scenes/stairs/seq-01/")
    
    print("Loading 100 frames...")
    frames = load_frames_from_sequence(seq_dir, num_frames=100)  # (100, 3, H, W)
    print(f"Loaded frames shape: {frames.shape}")
    
    # 移动到GPU（如果有）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames = frames.to(device)
    
    # 方法1: FFT法
    print("\nComputing FFT clarity...")
    clarity_fft = clarity_reward_from_image_fft(frames)
    clarity_fft = clarity_fft.cpu()
    print(f"FFT clarity range: [{clarity_fft.min():.3f}, {clarity_fft.max():.3f}]")
    print(f"FFT clarity variance: {clarity_fft.var():.3f}")
    
    # 方法2: 梯度法
    print("\nComputing Gradient clarity...")
    clarity_grad = clarity_reward_from_image(frames)
    clarity_grad = clarity_grad.cpu()
    print(f"Gradient clarity range: [{clarity_grad.min():.3f}, {clarity_grad.max():.3f}]")
    print(f"Gradient clarity variance: {clarity_grad.var():.3f}")
    
    # 可视化对比
    print("\nVisualizing results...")
    visualize_clarity_results(frames.cpu(), clarity_fft, "FFT Method")
    visualize_clarity_results(frames.cpu(), clarity_grad, "Gradient Method")
    
    # 打印最清晰/最模糊的帧索引
    print("\n=== FFT Method ===")
    print("Top 5 clearest frames:", torch.topk(clarity_fft, 5).indices.tolist())
    print("Top 5 blurriest frames:", torch.topk(clarity_fft, 5, largest=False).indices.tolist())
    
    print("\n=== Gradient Method ===")
    print("Top 5 clearest frames:", torch.topk(clarity_grad, 5).indices.tolist())
    print("Top 5 blurriest frames:", torch.topk(clarity_grad, 5, largest=False).indices.tolist())
    
    # 计算两种方法的相关性
    correlation = torch.corrcoef(torch.stack([clarity_fft, clarity_grad]))[0, 1]
    print(f"\nCorrelation between methods: {correlation:.3f}")

if __name__ == "__main__":
    test_clarity_on_real_data()