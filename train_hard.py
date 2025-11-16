import sys
import csv
from datetime import datetime
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Bernoulli
import numpy as np
import clip
import os, glob, shutil
import random
from PIL import Image
from tqdm import tqdm

from typing import Dict, List, Union

from data_utils import set_random_seed, load_sample_frames, read_image_sequences
from vis_utils import vis_rgb_maps
from config import parse_args
from controller import Controller
from frame_recon import SelectedFrameReconstructor, infer_sequence
from reward_utils import *


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


def train_controller(
    args, 
    selector, 
    optimizer, 
    reward, 
    baseline, 
    log_prob, 
    entropies,
    entropy_coeff
):
    reward = reward + entropy_coeff * entropies
    if baseline is None:
        baseline = reward
    else:
        baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * reward
        baseline = baseline.clone().detach()

    adv = reward - baseline
    print("[TRAINING] advantage", adv.item())
    loss = -log_prob * adv 
    loss -= entropy_coeff * entropies
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()
    if args.controller_grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.controller_grad_clip)
    #print("[TRAINING] head.grad", selector.head.weight.grad.norm().item())
    #print("[TRAINING] lstm.weight_ih_l0.grad", selector.lstm.weight_ih_l0.grad.norm().item())
    optimizer.step()
    old_params = {name: param.clone().detach() for name, param in selector.named_parameters()}

    delta_norms = {}
    for name, param in selector.named_parameters():
        if param.grad is not None:
            delta = param - old_params[name]
            delta_norms[name] = torch.norm(delta).item()
    if hasattr(selector, 'queries'):
        print("[TRAINING] queries.grad", selector.queries.grad.norm().item())
        print("[TRAINING] queries.grad (max/min/mean)",
            selector.queries.grad.max().item(),
            selector.queries.grad.min().item(),
            selector.queries.grad.mean().item())
    else:
        print("[TRAINING] head.grad", selector.head.weight.grad.norm().item())
        print("[TRAINING] lstm.weight_ih.grad", selector.lstm.weight_ih.grad.norm().item())
        print("[TRAINING] lstm.weight_hh.grad", selector.lstm.weight_hh.grad.norm().item())

    info = {
        "loss": loss.item(),
        "reward": reward.item(),
        "baseline": baseline.item(),
        "adv": adv.item(),
        "entropy": entropies.item(),
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
            
            
def write_log(log_name, save_dir, row, keep_idx=None):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_name)
    first_write = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if first_write:
            writer.writerow(
                ['epoch', 'step', 'reward', 'reward_avg100',
                 'keep_ratio', 'keep_avg100', 'sparse', 'sparse_avg100',
                 'entropy', 'loss', 'lr', 'keep_idx']
            )
        keep_idx_str = ','.join(map(str, keep_idx)) if keep_idx is not None else ''

        writer.writerow(row + [keep_idx_str])
        
        

def main(args):
    # Random seed setting
    device = args.device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    cudnn.benchmark = True
    set_random_seed(args.seed)
    
    seq_names = read_image_sequences(args.train_seqs)
    print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
          f"[INFO] Load {len(seq_names)} image sequences from {args.train_seqs}.")
    
    reconstructor = SelectedFrameReconstructor(args).to(device)
    reconstructor.eval()

    seq_inds = [i for i in range(len(seq_names))]
    #random.shuffle(seq_inds)
    for seq_ind in seq_inds:
        # path setting
        seq_name = os.path.split(os.path.split(seq_names[seq_ind])[0])[-1] # e.g. chess
        seq_label = os.path.split(seq_names[seq_ind])[-1]                  # e.g. seq-03
        
        if args.hard_ratio > 0.0:
            save_dir = os.path.join(args.save_dir,  seq_name + "-" + seq_label, 
                    "free-" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            save_dir = os.path.join(args.save_dir, seq_name + "-" + seq_label, 
                            "free-" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
        log_name = "log_" + datetime.now().strftime("%m%d-%H%M%S") + "_train.csv"

        start = time.time()
        indices, frames = load_sample_frames(
                seq_names[seq_ind], 
                frame_interval=args.frame_interval, 
                pil_mode=True,
                max_frames=args.max_frame_num
        )
        total_frames = len(frames)
        end = time.time()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] Loading images consumes {end - start:.2f} s.")
        
        # Selector
        sel_k = int(total_frames * args.hard_ratio)

        #selector = FrameController(total_frames, sel_k, args.feat_dim, 256).to(device)
        selector = Controller(
            args,                   
            feat_dim=args.feat_dim, 
            slot_sz=args.slot_sz, 
            seq_len=total_frames
        ).to(device)
        selector.train()
        optimizer = torch.optim.SGD(selector.parameters(), lr=args.controller_lr, momentum=0.9)

        # for reinforce learning
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

        # Get frame features of each sequence at begining
        start = time.time()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
            f"[INFO] Pass forward...")
        clip_model, preprocess = clip.load(
            '/opt/ml/code/cvpr2026/models/FrameSelector/pretrained/ViT-B-32.pt', 
            device=device
        )
        with torch.no_grad():
            preprocessed_images = [preprocess(image).to(args.device) for image in frames]
            stacked_images = torch.stack(preprocessed_images)
            frame_feats = clip_model.encode_image(stacked_images) # (S, 512)
            del preprocessed_images, stacked_images

        with torch.amp.autocast(device, enabled=True, dtype=dtype):
            #frame_feats = reconstructor.get_frame_feat(frames)
            full_preds, _ = reconstructor(frames)

        frame_feats = frame_feats.float()  # (S, 512)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=256, whiten=False)
        frame_feats = torch.from_numpy(
            pca.fit_transform(frame_feats.cpu().numpy())).to(device)
        
        frame_feats = frame_feats.unsqueeze(0)
        reconstructor.free_image_cache() # free images
        end = time.time()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] Forward consumes: {end - start:.2f} s.")

        with torch.no_grad():
            # L2 归一化
            feats = frame_feats / frame_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            #  cosine 矩阵
            cos_mat = feats.squeeze() @ feats.squeeze().T           # (S, S)
            # 屏蔽对角线
            mask = ~torch.eye(cos_mat.size(0), device=feats.device, dtype=torch.bool)
            cos_vec = cos_mat[mask]             # 一维，长度 S(S-1)

            mean = cos_vec.mean().item()
            std  = cos_vec.std().item()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
            f"[INFO] Cosine similarity between frames, Mean: {mean:.3f}, Std: {std:.3f}")
        
        # move to cpu for sample
        full_rgb_maps = full_preds["images"].detach().cpu()
        full_world_to_cam = full_preds["extrinsic"].detach().cpu()
        full_intrisic = full_preds["intrinsic"].detach().cpu()

        #full_cam_to_world = full_preds["cam_to_world"].detach().cpu

        # convert points cloud to voxel grids
        voxel_res = 128
        full_voxel_grid, bounds_min, bounds_max = voxelize_normalized_pcd(
            points=full_preds["world_points"],
            conf=full_preds["world_points_conf"],
            voxel_res=voxel_res,
            keep_ratio=0.9,
            debug=False,
        )

        # keep world points for later accuracy calculate
        full_pcd = full_preds["world_points"].cpu() # (N, 3)

        del full_preds # to clear gpu memory


        # CHECK IF IS DETERMINSTRIC
        history = set()
        stable_counter = 0
        patience = 20


        # reward EMA
        reward_ema = 0.0

        for epoch in tqdm(range(args.search_epochs)):
            #tau = max(0.5, args.temperature * (1 - epoch / args.search_epochs))
            entropy_coeff = max(1e-4, args.entropy_coeff * (1 - epoch / args.search_epochs))

            keep_idx, log_probs, entropies = selector.sample(frame_feats, args.temperature)

            keep_idx = sorted(keep_idx.tolist())
            keep_idx = [keep + i * args.slot_sz for i, keep in enumerate(keep_idx)]

            # first frame is reference
            is_add_ref = False
            #if 0 not in keep_idx:
            #    is_add_ref = True
            #    keep_idx = [0] + keep_idx

            # extract dropped indices
            all_idx = set(range(total_frames))
            keep_set = set(keep_idx)
            drop_idx = sorted(list(all_idx - keep_set))
            
            # sample uniformly
            max_render = 50
            n_drop     = len(drop_idx)
            slot_sz    = max(1, n_drop // max_render)
            slots      = [drop_idx[i:i+slot_sz] for i in range(0, n_drop, slot_sz)]
            sampled_drop_idx = []
            for s in slots:
                if len(s) == 0:
                    continue
                sampled_drop_idx.append(random.choice(s))
            if len(sampled_drop_idx) > max_render:
                sampled_drop_idx = sampled_drop_idx[::max(1, len(sampled_drop_idx)//max_render)]
            if not sampled_drop_idx:
                sampled_drop_idx = [drop_idx[0]] if drop_idx else [0]

            # selected frames
            sel_images = [frames[i] for i in keep_idx]

            # dropped pose
            extri = full_world_to_cam[sampled_drop_idx].to(device)
            intri = full_intrisic[sampled_drop_idx].to(device)

            # reconstruction and project using selected frames
            start = time.time()
            sel_rgb_map, sel_preds, *_ = infer_sequence(
                sel_images, 
                reconstructor,
                render_img_required=True, 
                embedding_required=False,
                pred_required=True,
                seq_size=args.infer_seq_size,
                pcd_conf_thresh=0.0,
                extrinsics=extri,
                intrinsics=intri
            )
            reconstructor.free_image_cache() # free images
            end = time.time()
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] Running {len(sel_images)} images after selection consumes {end - start:.2f} s.")
            
            # extract the world points
            sel_pcd = sel_preds["world_points"].detach().clone() # (sel_k, H, W, 3)
            sel_pcd_conf = sel_preds["world_points_conf"].detach().clone()
            del sel_preds

            if is_add_ref:
                # exclude the first ref frame
                sel_pcd = sel_pcd[1:]
                sel_pcd_conf = sel_pcd_conf[1:]
                #sel_rgb_map = sel_rgb_map[1:] 

            # extract pseudo-ground truth from global reconstruction results
            gt_rgb_map = full_rgb_maps[sampled_drop_idx].to(device) # (drop_k, 3, H, W)

            #----- DEBUG VIS
            #vis_rgb_maps(full_rgb_maps, os.path.join("./ckpt/vis/full", seq_name + "-" + seq_label), indices = [0, 1, 2])
            vis_rgb_maps(gt_rgb_map, os.path.join("./ckpt/vis/pseudo", seq_name + "-" + seq_label), indices = [0, 1, 2, -3, -2, -1])
            vis_rgb_maps(sel_rgb_map, os.path.join("./ckpt/vis/sparse", seq_name + "-" + seq_label), indices = [0, 1, 2, -3, -2, -1])
            #----- DEBUG VIS

            # Compute rewards
            alpha_l1 = 5.0
            alpha_u = 0.0
            alpha_c = 1.5
            alpha_acc = 1000

            l1 = -1.0 * alpha_l1 * F.l1_loss(sel_rgb_map, gt_rgb_map)
            uniformity = -1.0 * alpha_u * temporal_uniformity(keep_idx, total_frames)
            #contiguous = (torch.tensor(keep_idx[1:]) - torch.tensor(keep_idx[:-1])) <= 1
            #cont_len = contiguous.long().sum().float()
            #density_pen = -alpha_d * cont_len
            pcd_coverage = -1.0 * alpha_c * (1 - compute_pcd_coverage_reward(
                sel_pcd, sel_pcd_conf, full_voxel_grid,
                bounds_min, bounds_max, voxel_res, 
                sel_keep_ratio=1.0, debug=False
            ))
            acc, acc_med = accuracy_reward(full_pcd[keep_idx], sel_pcd)
            acc_reward = -1.0 * alpha_acc * (acc + acc_med)

            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] L1: {l1.item():.5f}"
                f" | Coverage: {pcd_coverage.item():.5f}"
                f" | Acc: {acc_reward:.5f}"
            )

            # norm
            #scale = np.sqrt(len(keep_idx)) + 1e-8
            #l1 /= scale
            #uniformity /= scale
            #pcd_coverage /= scale

            reward = l1 + uniformity + pcd_coverage + acc_reward
            
            sparse = torch.tensor(args.hard_ratio)

            baseline, train_info = train_controller(
                args, 
                selector, 
                optimizer, 
                reward, 
                baseline, 
                log_probs, 
                entropies,
                entropy_coeff
            )
            # Training information
            train_info["sparse"] = sparse.item()
            train_info["keep_ratio"] = len(keep_idx) / total_frames
            train_info["num_select"] = len(keep_idx)
            # monitor buffers
            reward_buf.append(train_info['reward'])
            sparse_buf.append(train_info['sparse'])
            kr_buf.append(train_info['keep_ratio'])


            reward_ema = 0.9 * reward_ema + 0.1 * reward.item()

            # Check if is determinstric
            #with torch.no_grad():
            #    logits = selector(frame_feats)
            #    determ_idx = logits.argmax(1).cpu().tolist()   
            #    _probs = logits.softmax(-1)
            #    top1_mean = _probs.max(-1)[0].mean().item()

            determ_idx, top1_mean = selector.inference(frame_feats)

            # train info print
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] "
                f"Epoch {epoch + 1} | "
                f"baseline ema: {train_info['baseline']:.4f} (reward avg100={sum(reward_buf)/len(reward_buf):.4f}) | "
                f"keep: {train_info['keep_ratio']:.2%} (avg100={sum(kr_buf)/len(kr_buf):.2%}) | "
                f"sparse: {train_info['sparse']:.4f} (avg100={sum(sparse_buf)/len(sparse_buf):.4f}) | "
                f"entropy: {train_info['entropy']:.6f} | "
                f"loss: {train_info['loss']:+.4f} | "
                f"top1 score: {top1_mean:.4e}"
            )
            
            determ_idx = tuple(sorted(determ_idx))
            
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
                train_info['lr'],
                determ_idx
            ])

            try:
                del gt_rgb_map, sel_rgb_map, sel_pcd, sel_pcd_conf
            except:
                pass

            torch.cuda.empty_cache()

            if determ_idx in history:
                stable_counter += 1
            else:
                stable_counter = 0               # 一旦新组合出现，清零
                history.add(determ_idx)


            # save checkpoints
            current_r = train_info['reward']
            is_best = current_r > best_reward_so_far
            if is_best:
                best_reward_so_far = current_r

            best_reward_record.append(reward)
            best_iter_record.append(epoch)

            state = {
                'epoch': epoch + 1,
                'controller_state': selector.state_dict(),
                'baseline': baseline,
                'reward': current_r,
                'args': args,
            }
            if (epoch + 1) % 20 == 0:
                fname = save_checkpoint(state, epoch + 1, current_r, save_dir, False)

            if is_best:
                fname = save_checkpoint(state, epoch + 1, current_r, save_dir, is_best)
            
            if stable_counter >= patience or top1_mean >= 0.98:
                print(f"[INFO] Early stop @ epoch {epoch + 1}  "
                    f"determinstric indices has been stable {patience} times: {list(determ_idx)}")
                fname = save_checkpoint(state, epoch + 1, current_r, save_dir, True)
                break

            if (epoch + 1) % 10 == 0:
                print(f"[INFO] Epoch {epoch + 1:03d}: Different counts={len(history)}")
                
            torch.cuda.empty_cache()

        try:
            del full_rgb_maps, frame_feats
        except:
            pass
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)