import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict


def compute_gae_advantages(
    rewards: torch.Tensor, 
    values: torch.Tensor,       
    #next_values: torch.Tensor,
    #masks: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns.

    :param rewards: Tensor of shape (B, K) representing rewards at each step.
    :param values: Tensor of shape (B, K + 1) representing value estimates at each state.
    :param gamma: Discount factor.
    :param lam: GAE parameter.

    :return: Tuple of (advantages, returns), both of shape (B, K).
    """
    B, K = rewards.shape
    device = rewards.device
    
    # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * values[:, 1:] - values[:, :-1]  # (B, K)

    # GAE advantage: A_t = Σ(γλ)^l δ_{t+l}
    advantages = torch.zeros_like(rewards)
    advantage = torch.zeros(B, device=device)
    
    for t in reversed(range(K)):
        advantage = deltas[:, t] + gamma * lam * advantage
        advantages[:, t] = advantage

    # normalize advantages, optional
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # returns: G_t = A_t + V(s_t)
    returns = advantages + values[:, :-1].detach()  # (B, K)

    return advantages, returns  # (B, K)


def a2c_loss(
    results: Dict[str, torch.Tensor],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the Actor-Critic loss.
    :param results: Output dictionary from the controller's forward pass.
    :param advantages: Tensor of shape (B, K) representing advantages.
    :param returns: Tensor of shape (B, K) representing returns.
    :param entropy_coeff: Coefficient for entropy regularization.
    :param value_coeff: Coefficient for value loss.

    :return: Tuple of (total_loss, loss_components)
    """
    log_probs = results['log_probs']      # (B, K)
    values    = results['values']         # (B, K+1)
    entropies = results['entropies']      # (B, K)

    # policy loss: -E[log π(a|s)·A]
    policy_loss = - (log_probs * advantages.detach()).mean()

    # value loss: E[(V(s) - G)^2]
    value_loss = F.mse_loss(values[:, :-1], returns.detach())

    # entropy loss: -E[H(π(·|s))]
    entropy_loss = -entropies.mean()

    total_loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

    return total_loss, {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy_loss': entropy_loss.item(),
        'entropy': entropies.mean().item()
    }


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
   # process dimension (S, 3, H, W)
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

    # construct high-frequency mask
    center = patch_size // 2
    Y, X = torch.meshgrid(
        torch.arange(patch_size, device=gray.device), 
        torch.arange(patch_size, device=gray.device), 
        indexing='ij'
    )
    radius = torch.sqrt((X - center)**2 + (Y - center)**2).float()
    mask = (radius > center * low_freq_radius_ratio).float()

    # compute high-frequency energy ratio
    high_freq_energy = (magnitude * mask).sum(dim=(-1, -2))
    total_energy = magnitude.sum(dim=(-1, -2)) + 1e-8
    high_freq_ratio = high_freq_energy / total_energy

    # reshape back to patch matrix for each frame (S, n_patches_h, n_patches_w)
    high_freq_ratio = high_freq_ratio.view(S, n_patches_h, n_patches_w)

    # spatial average: overall clarity for each frame (S,)
    clarity_per_frame = high_freq_ratio.mean(dim=(-1, -2))

    # sequence-wise normalization
    percentile_95 = torch.quantile(clarity_per_frame, 0.95)
    clarity_per_frame = clarity_per_frame / (percentile_95 + 1e-8)
    clarity_per_frame = torch.clamp(clarity_per_frame, 0.0, 1.0)
    
    return clarity_per_frame


def rotation_angle_from_matrix(R: torch.Tensor) -> float:
    """
    Compute rotation angle (in radians) from a rotation matrix.
    trace(R) = 1 + 2*cos(theta)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = (trace - 1) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angle = torch.acos(cos_theta)
    return angle.item()


def diversity_reward(
    action: int,
    selected_actions: List[int],
    camera_poses: torch.Tensor,
    neighbor_sz: int = 10,
    min_rot_angle_deg: float = 5.0,
 ) -> float:
    """
    Compute diversity reward based on camera pose differences.

    :param action: Current action (frame index).
    :param selected_actions: List of selected frame indices.
    :param camera_poses: Tensor of shape (S, 3, 4) representing camera poses (rotation matrix + translation).
    :param neighbor_sz: Number of nearest neighbors to consider for thresholding.
    :param min_rot_angle_deg: Minimum rotation angle distance (in degrees) for full rotation reward
    """

    if not selected_actions:
        return 1.0  # no penalty if no frames selected
    
   
    translations = camera_poses[:, :, 3]  # (S, 3)
    current_trans = translations[action]  # (3,)

    # compute L2 distances to all frames
    distances = torch.norm(
        current_trans.unsqueeze(0) - translations, 
        dim=-1, p=2
    )  # (S,)

    # get threshold distance (10th smallest distance)
    # small than it, may be redundant
    k = min(neighbor_sz, len(distances) - 1)
    threshold = torch.kthvalue(distances, k).value.item()

    # compute minimum distance to selected frames
    if selected_actions:
        selected_trans = translations[selected_actions]
        min_dist_to_selected = torch.norm(
            current_trans.unsqueeze(0) - selected_trans,
            dim=-1, p=2
        ).min().item()
    else:
        min_dist_to_selected = float('inf')

    # if beyond threshold, full reward
    if min_dist_to_selected > threshold:
        trans_reward = 1.0
    else:
        trans_reward = min_dist_to_selected / (threshold + 1e-8)

    if selected_actions:
        current_rot = camera_poses[action, :, :3]  # (3, 3)
        selected_rot = camera_poses[selected_actions, :, :3] #(K, 3, 3)

        rot_distances = []
        for sel_rot in selected_rot:
            R_rel = current_rot @ sel_rot.T  # rotation relative to selected frame
            angle = rotation_angle_from_matrix(R_rel) # degrees
            rot_distances.append(angle)

        rot_distances = torch.tensor(rot_distances)
        min_rot_rad = rot_distances.min().item()
    else:
        min_rot_rad = 3.14159  # the first selection gets full reward


    # rotation reward
    min_rot_deg = torch.rad2deg(torch.tensor(min_rot_rad)).item()
    rot_reward = min(min_rot_deg / min_rot_angle_deg, 1.0) # only greater than threshold gets full reward

    return 0.3 * trans_reward + 0.7 * rot_reward
