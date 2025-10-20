import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
from PIL import Image

from typing import Dict, List, Union

from data_utils import load_sample_frames, load_and_preprocess_sample_frames
from config import parse_args
from controller import FrameSelector
from frame_recon import SelectedFrameReconstructor


def set_random_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image_sequence_names(seq_folder:str)->List[str]:
    """ Read image seqences' names. """
    if not os.path.exists(seq_folder) or not os.path.isdir(seq_folder):
        raise FileNotFoundError(f"[Error] {seq_folder} is not found or is not a folder. ")
    seq_names = sorted([os.path.join(seq_folder, seq_name) \
                 for seq_name in os.listdir(seq_folder)])
    return seq_names


def infer_sequence(
    image_seq: List[Image.Image],
    reconstructor: SelectedFrameReconstructor,
    embedding_required=False
):
    """
    Given a list of PIL images, infer the reconstructed results using the reconstructor.
    :param image_seq: List of PIL images.
    :param reconstructor: The SelectedFrameReconstructor model.
    :param embedding_required: Whether to return the frame embeddings.
    :return rgb_map:  The projection of point clouds to RGB map tensor.
    :return pred_dict: Dictionary containing reconstruction results.
    :return embedding: (Optional) The frame embeddings from the teacher model.
    """
    with torch.no_grad():
        pred_dict, embedding = reconstructor(image_seq)

    with torch.no_grad():
        rgb_map, depth_map, conf_map, mask_map = reconstructor._project_world_points_to_images(
            pred_dict["images"], 
            pred_dict["world_points"], 
            pred_dict["world_points_conf"], 
            pred_dict["extrinsic"], 
            pred_dict["intrinsic"])
        
    if embedding_required:
        embedding = embedding.detach().requires_grad_()
        return rgb_map, pred_dict, embedding
    else:
        return rgb_map, pred_dict


def compute_reward(drop_render, gt_render):
    return -F.l1_loss(drop_render, gt_render)


def compute_reward_neighbor(
    drop_render: torch.Tensor, 
    gt_render: torch.Tensor, 
    keep_idx: torch.Tensor, 
    window_size: int=2
):
    """
    Compute reward based on local neighborhood L1 loss.
    :param drop_render: (S, 3, H, W) Rendered images from selected frames.
    :param gt_render: (S, 3, H, W) Pseudo-ground truth RGB images.
    :param keep_idx: (K,) Indices of selected frames.
    :param window_size: int Size of the neighborhood window on each side.
    :return: Mean reward over selected frames based on neighborhood L1 loss.
    """

    rewards = []
    device = drop_render.device

    for i in keep_idx:
        start = max(0, i - window_size)
        end   = min(gt_render.shape[0], i + window_size + 1)
        idx   = torch.arange(start, end, device=device)

        drop_render = drop_render[idx]          # (w,3,H,W)
        gt_render   = gt_render[idx]

        rewards.append(-F.l1_loss(drop_render, gt_render))

    return torch.stack(rewards).mean()


def select_topk(logits, k=None, ratio=None, hard=True):
    """
    logits: (B, S)  0~1概率
    k: int          固定选k帧
    ratio: float    按比例选
    hard: bool      True->返回0/1 mask，False->返回soft topk权重
    return:
        mask: (B, S)  0/1 或 soft权重
        log_prob: (B,)  选中的logits之和（用于RL）
    """
    B, S = logits.shape
    device = logits.device

    if k is None and ratio is not None:
        k = max(1, int(ratio * S))
    else:
        k = k or max(1, int(0.1 * S))

    # 取top-k索引
    scores = torch.sigmoid(logits)
    _, top_idx = torch.topk(scores, k, dim=1)  # (B, k)

    if hard:
        # 0/1 mask
        mask = torch.zeros_like(logits)
        mask.scatter_(1, top_idx, 1.0)
        # 梯度用straight-through：forward=0/1，backward=scores
        mask = (mask - scores).detach() + scores
    else:
        # soft：只保留top-k权重，其余置0
        mask = torch.zeros_like(scores)
        mask.scatter_(1, top_idx, scores.gather(1, top_idx))
        # 归一化到总和=k（可选）
        mask = mask * k / (mask.sum(dim=1, keepdim=True) + 1e-6)

    # 简单log_prob：选中帧的logits之和
    log_prob = logits.gather(1, top_idx).sum(dim=1)
    return mask, log_prob


def train_controller(args, selector, optimizer, reward, baseline, log_prob, entropy):
    selector.train()
    if baseline is None:
        baseline = reward
    else:
        baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * reward
        baseline = baseline.clone().detach()

    adv = reward - baseline
    loss = -log_prob * adv - args.entropy_coeff * entropy
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()
    if args.controller_grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.controller_grad_clip)
    optimizer.step()

    info = {
        "loss": loss.item(),
        "reward": reward.item(),
        "baseline": baseline.item(),
        "adv": adv.item(),
        "entropy": entropy.item(),
        "lr": optimizer.param_groups[0]['lr'],
    }
    return baseline, info


def main(args):
    # Random seed setting
    device = args.device
    cudnn.benchmark = True
    set_random_seed(args.seed)

    reconstructor = SelectedFrameReconstructor(args).to(device)
    reconstructor.eval()
    selector = FrameSelector(args, feat_dim=args.feat_dim).to(device)
    optimizer = torch.optim.SGD(selector.parameters(), lr=args.controller_lr, momentum=0.9)

    seq_names = load_image_sequence_names(args.data_root)
    print(f"[Info] Found {len(seq_names)} image sequences from {args.data_root}.")

    baseline = None

    best_reward_record, best_iter_record = [], []
    from collections import deque
    reward_buf = deque(maxlen=100)
    sparse_buf = deque(maxlen=100)
    kr_buf     = deque(maxlen=100)      # keep ratio buffer

    for epoch in range(args.search_epochs):
        print(f"\n===== Epoch {epoch + 1}/{args.search_epochs} =====")

        seq_path = np.random.choice(seq_names)
        print(seq_path)

        indices, frames = load_sample_frames(seq_path, frame_interval=args.frame_interval, pil_mode=True)

        gt_rgb_map, gt_pred_dict, embedding = infer_sequence(frames, reconstructor, embedding_required=True)
        logits, _ = selector(embedding)

        mask, log_prob, entropy = selector.sample(logits, temp=args.temperature, hard=False)

        # top-k or top-ratio selection, no entropy
        #if args.use_ratio:
        #    mask, log_prob = select_topk(logits, ratio=args.select_ratio, hard=True)
        #else:
        #   mask, log_prob = select_topk(logits, k=args.select_k, hard=True)
        #entropy = torch.zeros_like(log_prob)  # no entropy term if using hard selection

        # drop the frame which scores less than 0.5
        keep_idx = torch.where(mask.squeeze() > 0.5)[0].cpu().numpy()
        if keep_idx.size == 0:                       # if none selected, select the top-1
            _, top_idx = torch.topk(mask, k=1)
            keep_idx = [top_idx.item()]
        sel_images = [frames[i] for i in keep_idx]

        # 4. reconstruction and project with selected frames
        dropped_rgb_map, _ = infer_sequence(sel_images, reconstructor)

        # Reward computation
        sparse = 1.0 - mask.mean()
        reward = compute_reward(dropped_rgb_map, gt_rgb_map[keep_idx]) + \
            args.entropy_coeff * entropy + args.sparse_coeff * sparse

        baseline, train_info = train_controller(args, selector, optimizer, reward, baseline, log_prob, entropy)

        # training information
        train_info["sparse"] = sparse.item()
        train_info["keep_ratio"] = (mask > 0.5).float().mean().item()
        train_info["num_select"] = len(keep_idx)
        # monitor buffers
        reward_buf.append(train_info['reward'])
        sparse_buf.append(train_info['sparse'])
        kr_buf.append(train_info['keep_ratio'])

        # train info print
        print(f"[INFO] Epoch {epoch + 1} | "
            f"reward={train_info['reward']:+.4f} (avg100={sum(reward_buf)/len(reward_buf):+.4f}) | "
            f"keep={train_info['keep_ratio']:.2%} (avg100={sum(kr_buf)/len(kr_buf):.2%}) | "
            f"sparse={train_info['sparse']:.4f} (avg100={sum(sparse_buf)/len(sparse_buf):.4f}) | "
            f"entropy={train_info['entropy']:.3f} | "
            f"loss={train_info['loss']:+.4f} | "
            f"lr={train_info['lr']:.2e}"
        )


        best_reward_record.append(reward)
        best_iter_record.append(epoch)

        
if __name__ == "__main__":
    args = parse_args()
    main(args)