import sys
import csv
from datetime import datetime
import time
import os
import random
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import cv2

import clip

from config import parse_args
from models.data_utils import read_image_sequences, load_sample_frames, set_random_seed
from models.controller import Controller
from models.loss import *

from typing import Dict, List, Union


def build_optimizer(args, controller: Controller):
    """Build optimizer for controller model."""
    optimizer = torch.optim.Adam([
        {
            'params': [controller.start_token], 
            'lr': 0.1 * args.controller_lr,
            'weight_decay': 0.0,
        },
        {
            'params': 
                list(controller.encoder.parameters()) + 
                list(controller.decoder.parameters()) +
                list(controller.W_q.parameters()) +
                list(controller.W_h.parameters()), 
            'lr': args.controller_lr,
            'weight_decay': args.weight_decay,
        },
        {
            'params': [controller.value_head],
            'lr': 1.5 * args.controller_lr,
            'weight_decay': 10.0 * args.weight_decay,
        }
    ])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=300,
        gamma=0.8
    )
    return optimizer, lr_scheduler


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
    for seq_ind in seq_inds:
        # Path setting
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
        
        max_select_nums = max(1, int(args.select_ratio * total_frames))
        controller = Controller(
            feat_size=args.feat_size,
            hidden_size=args.controller_hid_size,
            max_select_nums=max_select_nums
        )

        # Load warmup checkpoint if exists
        if args.warmup_ckpt_path is not None and os.path.exists(args.warmup_ckpt_path):
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] Load warm-up checkpoint from {args.warmup_ckpt_path}.")
            checkpoint = torch.load(args.warmup_ckpt_path, map_location='cpu')
            controller.load_state_dict(checkpoint['controller_state'])

        controller = controller.to(device)

        # Build optimizer and learning rate scheduler
        optimizer, lr_scheduler = build_optimizer(args, controller)

        # CLIP forward to get frame features
        start = time.time()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
            f"[INFO] CLIP forward...")
        clip_model, preprocess = clip.load(
            '/opt/ml/code/cvpr2026/models/FrameSelector/pretrained/ViT-B-32.pt', 
            device=device
        )
        with torch.no_grad():
            preprocessed_images = [preprocess(image).to(args.device) for image in frames]
            stacked_images = torch.stack(preprocessed_images)
            frame_feats = clip_model.encode_image(stacked_images).unsqueeze(0) # (1, S, 512)
            del preprocessed_images, stacked_images
        frame_feats = frame_feats.float()  # (1, S, 512)
        end = time.time()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] CLIP consumes: {end - start:.2f} s.")

        # VGGT forward to get predictions
        start = time.time()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
            f"[INFO] VGGT forward...")
        with torch.amp.autocast(device, enabled=True, dtype=dtype):
            #frame_feats = reconstructor.get_frame_feat(frames)
            full_preds, _ = reconstructor(frames)
        reconstructor.free_image_cache() # free images   
        end = time.time()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
            f"[INFO] VGGT consumes: {end - start:.2f} s.")       

        # Prepare rewards
        clarity_rewards = clarity_reward_from_image_fft(
            rgb_image=full_preds['images'],
            low_freq_radius_ratio=0.08,
            patch_size=32,
            pad_mode='reflect'
        ) # (S,)
        
        # VGGT predictions as pseudo-ground truth
        full_intrisics = full_preds['intrinsics'].detach().clone() # (S, 3, 3)
        full_cam_to_worlds = full_preds['cam_to_worlds'].detach().clone() # (S, 3, 4)

        del full_preds # save memory

        # Training loop for controller
        controller.train()
        for epoch in tqdm(range(args.search_epochs)):
            
            results = controller(
                frame_feats, 
                temperature=args.temperature
            )

            actions = results['actions']      # (1, K)
            values = results['values']        # (1, K+1)

            keep_inds = actions[0].cpu().numpy().tolist()
            #selected_frames = [frames[i] for i in keep_inds]

            # compute marginal rewards
            selected_inds = []
            sel_diversity_rewards = []
            for sel_ind in keep_inds:
                div_reward = diversity_reward(
                    action=sel_ind,
                    selected_actions=selected_inds,
                    camera_poses=full_cam_to_worlds,
                    trans_percentile=0.1 * args.select_ratio,
                    min_rot_angle_deg=5.0,
                )
                selected_inds.append(sel_ind)
                sel_diversity_rewards.append(div_reward)

            # both rewards are in 0-1 scale, shape of (K,)
            sel_diversity_rewards = torch.cat(sel_diversity_rewards, dim=0)
            sel_clarity_rewards = clarity_rewards[keep_inds]
            rewards = 0.5 * sel_diversity_rewards + 0.5 * sel_clarity_rewards

            # compute advantages and returns
            advantages, returns = compute_gae_advantages(
                rewards=rewards,
                values=values[0], # (K+1,)
                gamma=0.99,
                lam=0.95
            )

            # compute loss
            loss, loss_components = a2c_loss(
                results=results,
                advantages=advantages,
                returns=returns,
                entropy_coeff=args.entropy_coeff,
                value_coeff=args.value_coeff
            )

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            if args.controller_grad_clip > 0.0:
                nn.utils.clip_grad_norm_(
                    controller.parameters(), 
                    args.controller_grad_clip
                )
            optimizer.step()
            lr_scheduler.step()

            # logging
            print(
                f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] loss: {loss.item():.4f}, "
                f"policy loss: {loss_components['policy_loss']:.4f}, "
                f"value loss: {loss_components['value_loss']:.4f}, "
                f"entropy: {loss_components['entropy']:.4f}, "
                f"clarity reward: {sel_clarity_rewards.mean().item():.4f}, "
                f"diversity reward: {sel_diversity_rewards.mean().item():.4f}"
            )

            torch.cuda.empty_cache()