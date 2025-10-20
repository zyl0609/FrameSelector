import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import os, glob, shutil
import random
from PIL import Image
from tqdm import tqdm

from typing import Dict, List, Union

from data_utils import load_sample_frames, read_image_sequences
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


def load_image_sequences(seq_folder:str)->List[str]:
    """ Read image seqences' names. """
    if not os.path.exists(seq_folder) or not os.path.isdir(seq_folder):
        raise FileNotFoundError(f"[Error] {seq_folder} is not found or is not a folder. ")
    
    seq_paths = []
    for seq_name in os.listdir(seq_folder):
        seq_path = os.path.join(seq_folder, seq_name)
        if os.path.isdir(seq_path):
            seq_paths.append(seq_path)
    seq_paths = sorted(seq_paths)
    return seq_paths


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
        return rgb_map, embedding
    else:
        return rgb_map, None


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


def save_checkpoint(state, epoch, reward, save_dir, is_best=False):
    """
    Save the training state to a checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    # 主文件
    fname = os.path.join(save_dir, f"ckp_ep{epoch:04d}_r{reward:.4f}.pth")
    torch.save(state, fname)
    # best 软链接
    if is_best:
        best_link = os.path.join(save_dir, "best.pth")
        # 先删旧链接
        if os.path.islink(best_link) or os.path.exists(best_link):
            os.remove(best_link)
        os.symlink(os.path.basename(fname), best_link)
    return fname


def cleanup_checkpoints(ckp_queue, max_ckp):
    """Delete old checkpoints to save disk space."""
    while len(ckp_queue) > max_ckp:
        worst_r, worst_ep, worst_path = ckp_queue.pop(0)
        if os.path.exists(worst_path):
            os.remove(worst_path)
            print(f"[CLEAN] removed ckp {worst_path}")


def main(args):
    # Random seed setting
    device = args.device
    cudnn.benchmark = True
    set_random_seed(args.seed)

    reconstructor = SelectedFrameReconstructor(args).to(device)
    reconstructor.eval()
    selector = FrameSelector(args, feat_dim=args.feat_dim).to(device)
    optimizer = torch.optim.SGD(selector.parameters(), lr=args.controller_lr, momentum=0.9)

    seq_names = read_image_sequences(args.train_seqs)
    print(f"[INFO] Load {len(seq_names)} image sequences from {args.train_seqs}.")

    baseline = None

    # records
    best_reward_record, best_iter_record = [], []
    ckp_queue = []
    max_ckp = 3
    best_reward_so_far = -float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Buffer for printing running averages
    from collections import deque
    reward_buf = deque(maxlen=100)
    sparse_buf = deque(maxlen=100)
    kr_buf     = deque(maxlen=100)      # keep ratio buffer

    for epoch in range(args.search_epochs):
        print(f"\n===== Epoch {epoch + 1}/{args.search_epochs} =====")

        seq_path = np.random.choice(seq_names)
        print(seq_path)

        indices, frames = load_sample_frames(seq_path, frame_interval=args.frame_interval, pil_mode=True)
        # TODO: limit max frames
        if len(frames) > args.max_frame_num:
            # randomly sample a subset of frames
            selected_indices = sorted(np.random.choice(len(frames), args.max_frame_num, replace=False))
            frames = [frames[i] for i in selected_indices]
            indices = [indices[i] for i in selected_indices]


        gt_rgb_map, embedding = infer_sequence(frames, reconstructor, embedding_required=True)
        logits, _ = selector(embedding)

        mask, log_prob, entropy = selector.sample(logits, temp=args.temperature, hard=False)

        # top-k or top-ratio selection, no entropy
        #if args.use_ratio:
        #    mask, log_prob = select_topk(logits, ratio=args.select_ratio, hard=True)
        #else:
        #   mask, log_prob = select_topk(logits, k=args.select_k, hard=True)
        #entropy = torch.zeros_like(log_prob)  # no entropy term if using hard selection

        # Drop the frame which scores less than 0.5
        keep_idx = torch.where(mask.squeeze() > 0.5)[0].cpu().numpy()
        if keep_idx.size == 0:                       # if none selected, select the top-1
            _, top_idx = torch.topk(mask, k=1)
            keep_idx = [top_idx.item()]
        sel_images = [frames[i] for i in keep_idx]

        # Reconstruction and project with selected frames
        dropped_rgb_map, _ = infer_sequence(sel_images, reconstructor)

        # Reward computation
        sparse = 1.0 - mask.mean()
        reward = compute_reward(dropped_rgb_map, gt_rgb_map[keep_idx]) + args.sparse_coeff * sparse

        baseline, train_info = train_controller(args, selector, optimizer, reward, baseline, log_prob, entropy)

        # Training information
        train_info["sparse"] = sparse.item()
        train_info["keep_ratio"] = (mask > 0.5).float().mean().item()
        train_info["num_select"] = len(keep_idx)
        # monitor buffers
        reward_buf.append(train_info['reward'])
        sparse_buf.append(train_info['sparse'])
        kr_buf.append(train_info['keep_ratio'])

        # train info print
        print(f"[INFO] Epoch {epoch + 1} | "
            f"reward: {train_info['reward']:+.4f} (avg100={sum(reward_buf)/len(reward_buf):+.4f}) | "
            f"keep: {train_info['keep_ratio']:.2%} (avg100={sum(kr_buf)/len(kr_buf):.2%}) | "
            f"sparse: {train_info['sparse']:.4f} (avg100={sum(sparse_buf)/len(sparse_buf):.4f}) | "
            f"entropy: {train_info['entropy']:.3f} | "
            f"loss: {train_info['loss']:+.4f} | "
            f"learning rate: {train_info['lr']:.2e}"
        )

        # save the best checkpoint
        current_r = train_info['reward']
        is_best = current_r > best_reward_so_far
        if is_best:
            best_reward_so_far = current_r

        # save the current checkpoint
        state = {
            'epoch': epoch + 1,
            'controller_state': selector.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'baseline': baseline,
            'reward': current_r,
            'args': args,
        }
        fname = save_checkpoint(state, epoch + 1, current_r, args.save_dir, is_best)

        # maintain ckp queue
        ckp_queue.append((current_r, epoch + 1, fname))
        ckp_queue.sort(key=lambda x: x[0])          # 小顶堆
        cleanup_checkpoints(ckp_queue, max_ckp)

        best_reward_record.append(reward)
        best_iter_record.append(epoch)

        torch.cuda.empty_cache()
        

        
if __name__ == "__main__":
    args = parse_args()
    main(args)