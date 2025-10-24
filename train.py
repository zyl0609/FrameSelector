import sys
import csv
from datetime import datetime
import time
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
from frame_recon import SelectedFrameReconstructor, infer_sequence
from reward_utils import ssim_loss
from evaluate import evaluate_pcd


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


def compute_reward(drop_render, gt_render, alpha=0.5, window_size=11):
    l1 = F.l1_loss(drop_render, gt_render).item()
    ssim = ssim_loss(drop_render, gt_render, window_size=window_size).item()
    reward = -(alpha * l1 + (1 - alpha) * ssim)
    print("[TRAINING] l1 =", l1)
    print("[TRAINING] ssim =", ssim)
    return reward


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
    print("[TRAINING] head.grad", selector.head.weight.grad.norm().item())
    print("[TRAINING] lstm.weight_ih.grad", selector.lstm.weight_ih.grad.norm().item())
    
    old_params = {name: param.clone().detach() for name, param in selector.named_parameters()}

    optimizer.step()
    
    delta_norms = {}
    for name, param in selector.named_parameters():
        if param.grad is not None:
            delta = param - old_params[name]
            delta_norms[name] = torch.norm(delta).item()
    print(f"[UPDATE] head.weight Δ = {delta_norms.get('head.weight', 0):.6f}")
    print(f"[UPDATE] lstm.weight_ih Δ = {delta_norms.get('lstm.weight_ih', 0):.6f}")

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
    fname = os.path.join(save_dir, f"ckp_ep_{epoch:04d}_reward_{reward:.4f}.pth")
    torch.save(state, fname)
    if is_best:
        best_link = os.path.join(save_dir, "best.pth")
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
            #print(f"[CLEAN] removed ckp {worst_path}")
            
            
def write_log(log_name, save_dir, row):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_name)
    first_write = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if first_write:
            writer.writerow(
                ['epoch', 'step', 'reward', 'reward_avg100',
                 'keep_ratio', 'keep_avg100', 'sparse', 'sparse_avg100',
                 'entropy', 'loss', 'lr']
            )
        writer.writerow(row)
        
        

            

def main(args):
    # path setting
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_name = "log_" + datetime.now().strftime("%m%d-%H%M%S") + "_train.csv"


    # Random seed setting
    device = args.device
    cudnn.benchmark = True
    set_random_seed(args.seed)
    
    
    reconstructor = SelectedFrameReconstructor(args).to(device)
    reconstructor.eval()
    selector = FrameSelector(args, feat_dim=args.feat_dim).to(device)
    optimizer = torch.optim.SGD(selector.parameters(), lr=args.controller_lr, momentum=0.9)
    
    #from debugger import ParamInspector
    #inspector = ParamInspector(selector)
    

    seq_names = read_image_sequences(args.train_seqs)
    print(f"[INFO] Load {len(seq_names)} image sequences from {args.train_seqs}.")

    baseline = None

    # records
    best_reward_record, best_iter_record = [], []
    ckp_queue = []
    max_ckp = 10
    best_reward_so_far = -float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Buffer for printing running averages
    from collections import deque
    reward_buf = deque(maxlen=100)
    sparse_buf = deque(maxlen=100)
    kr_buf     = deque(maxlen=100)      # keep ratio buffer

    for epoch in range(args.search_epochs):
        seq_inds = [i for i in range(len(seq_names))]
        random.shuffle(seq_inds)
        
        for seq_ind in tqdm(seq_inds):
            indices, frames = load_sample_frames(
                seq_names[seq_ind], 
                frame_interval=args.frame_interval, 
                pil_mode=True,
                max_frames=args.max_frame_num
            )
            start = time.time()
            # only embedding needed
            _, _, embeddings = infer_sequence(
                frames, 
                reconstructor, 
                render_img_required=False, 
                embedding_required=True,
                seq_size=args.infer_seq_size,
                pcd_conf_thresh=args.pcd_conf_thresh
            )
            reconstructor.free_image_cache() # free images
            end = time.time()
            #print(f"[INFO] Running {len(indices)} images consumes {end - start} s.")
            print(f"[INFO] Getting embeddings consumes {end - start:.2f} s.")
            
            print(f"[INFO] Selecting key frames...")
            logits, _ = selector(embeddings)
            mask, log_prob, entropy = selector.sample(logits, temp=args.temperature, hard=False)
            keep_idx = torch.where(mask.squeeze() > 0.5)[0].cpu().numpy()
            if keep_idx.size == 0:                       # if none selected, select the top-1
                _, top_idx = torch.topk(mask, k=1)
                keep_idx = [top_idx.item()]
            sel_images = [frames[i] for i in keep_idx]
            
            start = time.time()
            neighbor_sz = args.vggt_neighbor_size
            gt_rgb_list = []
            for i in keep_idx:                            # keep_idx 是 List[int]
                left  = max(0, i - neighbor_sz)
                right = min(len(frames), i + neighbor_sz + 1)
                nb_idx = np.arange(left, right)
                 
                valid_nb = np.intersect1d(nb_idx, keep_idx)
                if len(valid_nb) < 3:                      # at least 3 frame to reconstruct
                    delta = 3 - len(valid_nb)
                    left  = max(0, left - delta)
                    right = min(len(frames), right + delta)
                    valid_nb = np.intersect1d(np.arange(left, right), keep_idx)
                local_imgs = [frames[int(idx)] for idx in valid_nb]

                local_rgb_map, _, _ = infer_sequence(
                    local_imgs, 
                    reconstructor,
                    pcd_conf_thresh=args.pcd_conf_thresh
                ) #(L, 3, H, W)
                
                reconstructor.free_image_cache() # free images
                local_idx = np.where(valid_nb == i)[0].item()   # current frame's index in neighborhood
                local_rgb = local_rgb_map[local_idx].detach().clone()    # (3, H, W)
                del local_rgb_map
                gt_rgb_list.append(local_rgb)

            gt_rgb_map = torch.stack(gt_rgb_list, dim=0).to(device) # (K, 3, H, W)
            del gt_rgb_list
            end = time.time()
            print(f"[INFO] Reconstruction local neighbors consumes: total {end - start:.2f} s, average {(end - start) / len(keep_idx):.2f} s")

            # Reconstruction and project with selected frames
            start = time.time()
            dropped_rgb_map, _, _ = infer_sequence(
                sel_images, 
                reconstructor, 
                seq_size=args.infer_seq_size,
                pcd_conf_thresh=args.pcd_conf_thresh
            )
            reconstructor.free_image_cache() # free images
            end = time.time()
            print(f"[INFO] Running {len(sel_images)} images after selection consumes {end - start:.2f} s.")
            
            
            #----- DEBUG VIS
            from data_utils import vis_rgb_maps
            seq_label = os.path.split(seq_names[seq_ind])[-1]
            seq_name = os.path.split(os.path.split(seq_names[seq_ind])[0])[-1]
            vis_rgb_maps(gt_rgb_map, os.path.join("./ckpt/vis/pseudo", seq_name + "-" + seq_label), indices = [0, 1, 2])
            vis_rgb_maps(dropped_rgb_map, os.path.join("./ckpt/vis/sparse", seq_name + "-" + seq_label), indices = [0, 1, 2])
            #----- DEBUG VIS
            
            
            # Reward computation
            sparse = 1.0 - torch.where(mask.squeeze() > 0.5)[0].mean()
            reward = compute_reward(dropped_rgb_map, gt_rgb_map) + args.sparse_coeff * sparse

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
            print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] "
                f"Epoch {epoch + 1} | "
                f"reward: {train_info['reward']:+.4f} (avg100={sum(reward_buf)/len(reward_buf):+.4f}) | "
                f"keep: {train_info['keep_ratio']:.2%} (avg100={sum(kr_buf)/len(kr_buf):.2%}) | "
                f"sparse: {train_info['sparse']:.4f} (avg100={sum(sparse_buf)/len(sparse_buf):.4f}) | "
                f"entropy: {train_info['entropy']:.3f} | "
                f"loss: {train_info['loss']:+.4f} | "
                f"lr: {train_info['lr']:.2e}"
            )
            
            write_log(
                log_name,
                save_dir, [
                epoch + 1,
                epoch * len(seq_inds) + seq_ind,
                train_info['reward'],
                sum(reward_buf) / len(reward_buf),
                train_info['keep_ratio'],
                sum(kr_buf) / len(kr_buf),
                train_info['sparse'],
                sum(sparse_buf) / len(sparse_buf),
                train_info['entropy'],
                train_info['loss'],
                train_info['lr']
            ])

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
            fname = save_checkpoint(state, epoch + 1, current_r, save_dir, is_best)

            # maintain ckp queue
            ckp_queue.append((current_r, epoch + 1, fname))
            ckp_queue.sort(key=lambda x: x[0])          # 小顶堆
            cleanup_checkpoints(ckp_queue, max_ckp)

            best_reward_record.append(reward)
            best_iter_record.append(epoch)

            del gt_rgb_map, embeddings, dropped_rgb_map, logits, mask, log_prob, entropy, reward
            torch.cuda.empty_cache()
            
            
        #if (epoch + 1) % args.val_epoch == 0:
        #    print("\n[INFO] Validate the selector...")
        #    selector.eval()
        #    print("[INFO] Running without selection...")
        #    evaluate_pcd(args, reconstructor, val_epoch=epoch+1)
        #    print("[INFO] Running with selection...")
        #    evaluate_pcd(args, reconstructor, selector, val_epoch=epoch+1)
        #    print("[INFO] Done...")
        #    selector.train()
            
        

        
if __name__ == "__main__":
    args = parse_args()
    main(args)