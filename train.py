from datetime import datetime
import time
import torch
import torch.nn as nn
import numpy as np
import clip
import os, glob, shutil
import random
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

from typing import Dict, List, Union

from config import parse_args
from utility.data_utils import set_random_seed, load_sample_frames, read_image_sequences
from utility.pc_utils import point_cloud_to_volume, icp, accuracy, completion, coverage
from models.frame_recon import SelectedFrameReconstructor
from models.controller import Controller


def main(args):
    # Random seed setting
    device = args.device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
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

        # CLIP forward
        start = time.time()
        clip_model, preprocess = clip.load(
            '/opt/ml/code/cvpr2026/models/FrameSelector/pretrained/ViT-B-32.pt', 
            device=args.device
        )
        with torch.no_grad():
            preprocessed_images = [preprocess(image).to(args.device) for image in frames]
            stacked_images = torch.stack(preprocessed_images)
            frame_feats = clip_model.encode_image(stacked_images) # (S, 512)
            del preprocessed_images, stacked_images

        frame_feats = frame_feats.float().unsqueeze(0)
        end = time.time()

        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] CLIP forward consumes: {end - start:.2f} s.")

        # K-means clustering on frame features
        # Reshape frame features for clustering: (1, S, 512) -> (S, 512)
        features_2d = frame_feats.squeeze(0).cpu().numpy()

        # Perform k-means clustering with 100 clusters
        n_clusters = min(100, len(frames))  # Ensure we don't have more clusters than frames
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_2d)

        # Create cluster mapping: cluster_id -> list of original frame indices
        cluster_to_indices = {}

        for i, label in enumerate(cluster_labels):
            if label not in cluster_to_indices:
                cluster_to_indices[label] = []
            cluster_to_indices[label].append(indices[i])  # Original frame indices
        
        controller = Controller(
            args,
            feat_size=args.feat_size,
            clusters=cluster_to_indices
        ).to(device)
        controller.train()
        optimizer = torch.optim.SGD(controller.parameters(), lr=args.controller_lr, momentum=0.9)

        # for ENAS
        baseline = None
        base_cov = None

        # VGGT forward
        start = time.time()
        with torch.amp.autocast(device, enabled=True, dtype=dtype):
            full_preds, _ = reconstructor(frames)
        reconstructor.free_image_cache() # free images

        end = time.time()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] VGGT forward consumes: {end - start:.2f} s.")
        
        # preprocess world points
        world_points = full_preds["world_points"].reshape(-1, 3) # (N, 3)
        world_points_conf = full_preds["world_points_conf"].reshape(-1)

        # baseline: uniform step
        step = total_frames // n_clusters
        uniform_indices = list(range(0, total_frames, step))
        uniform_frames = [frames[i] for i in uniform_indices]
        with torch.amp.autocast(device, enabled=True, dtype=dtype):
            uniform_preds, _ = reconstructor(uniform_frames)
        reconstructor.free_image_cache() # free images
        uniform_world_points = uniform_preds["world_points"].reshape(-1, 3)

        # filter points with low confidence
        pcd_keep_ratio = 0.9
        k = int(world_points_conf.size(0) * (1.0 - pcd_keep_ratio))
        conf_thresh = torch.kthvalue(world_points_conf, max(1, min(k, world_points_conf.size(0)))).values

        valid_mask = world_points_conf >= conf_thresh
        world_points = world_points[valid_mask]

        del full_preds, uniform_preds

        # downsample
        fused_points = point_cloud_to_volume(
            world_points,
            voxel_size=0.01
        )
        uniform_fused_points = point_cloud_to_volume(
            uniform_world_points,
            voxel_size=0.01
        )

        for epoch in tqdm(range(args.search_epochs)):
            entropy_coeff = max(1e-6, args.entropy_coeff * (1 - epoch / args.search_epochs))

            keep_idx, log_probs, entropies = controller.sample(frame_feats, args.temperature)
            keep_idx = sorted(keep_idx)

            sel_images = [frames[i] for i in keep_idx]
            start = time.time()
            with torch.amp.autocast(device, enabled=True, dtype=dtype):
                sel_preds, _ = reconstructor(sel_images)
            reconstructor.free_image_cache()
            end = time.time()
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] VGGT forward consumes: {end - start:.2f} s.")

            selected_world_points = sel_preds["world_points"].detach().clone()

            sel_fused_points = point_cloud_to_volume(
                selected_world_points,
                voxel_size=0.01
            )

            # Compute rewards using chamfer distance
            start = time.time()
            # runing ICP
            sel_points, full_points = icp(
                sel_fused_points, 
                fused_points, 
                threshold=0.025
            )
            # compute accuracy and completeness
            acc, acc_med = accuracy(
                full_points,
                sel_points
            )
            comp, comp_med = completion(
                full_points,
                sel_points
            )
            # compute propotion of inliers
            cov = coverage(
                full_points, 
                sel_points, 
                distance_threshold=0.01
            )
            end = time.time()
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] Compute Acc&Comp&Cov consumes: {end - start:.2f} s.")
            
            # compute baseline reward
            if base_cov is None:
                uniform_sel_points, _ = icp(
                    uniform_fused_points, 
                    fused_points, 
                    threshold=0.025
                )
                base_cov = coverage(
                    full_points,
                    uniform_sel_points,
                    distance_threshold=0.01
                )
                print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                    f"[INFO] Baseline Coverage: {base_cov:.3f}"
                )
            
            #reward = -1000.0 * (acc + comp) # chamfer distance
            #reward = 10.0 * cov
                
            alpha = 50.0 if cov > base_cov else 10.0
            reward = alpha * (cov - base_cov)

            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] Reward: {reward:.5f}, Cov: {cov:.3f}, Baseline:{base_cov:.3f}, Acc: {acc:.5f}, Comp: {comp:.5f}"
            )

            # update controller
            baseline, train_info = train_controller(
                args, 
                controller, 
                optimizer, 
                reward, 
                baseline, 
                log_probs, 
                entropies,
                entropy_coeff
            )

            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] "
                f"baseline: {train_info['baseline']:.4f} | "
                f"entropy: {train_info['entropy']:.6f} | "
                f"loss: {train_info['loss']:+.4f} | "
            )

            if (epoch + 1) % 10 == 0:
                selected_inds, _entropies = controller.inference(frame_feats)
                selected_inds = sorted(selected_inds)
                save_folder = os.path.join('log', seq_name, seq_label)
                os.makedirs(save_folder, exist_ok=True)
                save_path = os.path.join(save_folder, 'keep.txt')
                with open(save_path, 'a') as f:
                    f.write(f"#Epoch: {epoch+1}, Entropy: {_entropies:.6f}, Keep: {selected_inds}\n")

            try:
                del sel_preds, selected_world_points, sel_fused_points
            except:
                pass
            
        try:
            del frame_feats, world_points, world_points_conf, fused_points, uniform_fused_points
        except:
            pass
        
        torch.cuda.empty_cache()
                    
            
def train_controller(
    args, 
    controller, 
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

    #adv = reward - baseline
    adv = reward # directly use reward
    print("[TRAINING] advantage", adv.item())
    loss = -log_prob * adv 
    loss -= entropy_coeff * entropies
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()
    if args.controller_grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(controller.parameters(), args.controller_grad_clip)
    optimizer.step()

    print("[TRAINING] lstm.weight_ih.grad", controller.lstm.weight_ih.grad.norm().item())
    print("[TRAINING] lstm.weight_hh.grad", controller.lstm.weight_hh.grad.norm().item())

    info = {
        "loss": loss.item(),
        "reward": reward.item(),
        "baseline": baseline.item(),
        "adv": adv.item(),
        "entropy": entropies.item(),
        "lr": optimizer.param_groups[0]['lr'],
    }
    return baseline, info





if __name__ == "__main__":
    args = parse_args()
    main(args)