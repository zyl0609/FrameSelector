from datetime import datetime
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip

from config import *
from utility.data_utils import read_image_sequences, load_sample_frames, set_random_seed
from models.controller import Controller


def generate_uniform_teacher_actions(
        batch_size: int, 
        sequence_size: int, 
        max_select_nums: int, 
        device: torch.device
):
    """Generate equally spaced teacher actions for warm-up training."""
    K = min(sequence_size, max_select_nums)
    
    step = sequence_size // K
    
    # 0, step, 2*step, ..., (K-1)*step
    indices = torch.arange(0, K, device=device) * step
    
    return indices.unsqueeze(0).expand(batch_size, -1)


def main(args):
    set_random_seed(args.seed)

    device = args.device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    seq_names = read_image_sequences(args.train_seqs)
    print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
          f"[INFO] Load {len(seq_names)} image sequences from {args.train_seqs}.")
    
    seq_inds = [i for i in range(len(seq_names))]
    for seq_ind in seq_inds:
        seq_name = os.path.split(os.path.split(seq_names[seq_ind])[0])[-1] # e.g. chess
        seq_label = os.path.split(seq_names[seq_ind])[-1]                  # e.g. seq-03
        
        start = time.time()
        indices, frames = load_sample_frames(
                seq_names[seq_ind], 
                frame_interval=args.frame_interval, 
                pil_mode=True,
                max_frames=args.max_frame_num
        )
        total_frames = len(frames) # sequence length

        max_select_nums = max(1, int(args.select_ratio * total_frames))
        controller = Controller(
            feat_size=args.feat_size,
            hidden_size=args.controller_hid_size,
            max_select_nums=max_select_nums
        ).to(device)

        end = time.time()
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"[INFO] Loading images consumes {end - start:.2f} s.")
        
        # Get frame features of each sequence at begining
        start = time.time()
        clip_model, preprocess = clip.load(
            '/opt/ml/code/cvpr2026/models/FrameSelector/pretrained/ViT-B-32.pt', 
            device=device
        )
        with torch.no_grad():
            preprocessed_images = [preprocess(image).to(args.device) for image in frames]
            stacked_images = torch.stack(preprocessed_images)
            frame_feats = clip_model.encode_image(stacked_images) # (S, 512)
            del preprocessed_images, stacked_images

        frame_feats = frame_feats.unsqueeze(0).float()  # (1, S, 512)
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
            f"[INFO] Forward consumes: {end - start:.2f} s.")
        
        B, S, _ = frame_feats.shape

        optimizer = torch.optim.Adam([
            {'params': [controller.start_token], 'lr': 0.1 * args.warmup_lr},  # 小10倍
            {'params': list(controller.encoder.parameters()) + 
                        list(controller.decoder.parameters()) +
                        list(controller.W_q.parameters()) +
                        list(controller.W_h.parameters()), 'lr': args.warmup_lr},
        ])


        epoch_loss = 0.0
        for epoch in tqdm(range(100)):
            controller.train()
            teacher_actions = generate_uniform_teacher_actions(B, S, max_select_nums, device)

            results = controller(
                frame_feats, 
                teacher_actions=teacher_actions,
                teacher_forcing=True,
                temperature=2.0
            )

            logits = results['logits']

            loss = F.cross_entropy(
                logits.view(-1, S),       # (B*K, S)
                teacher_actions.view(-1)  # (B*K,)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.5)
            optimizer.step()

            # monitor training
            with torch.no_grad():
                controller.eval()

                _results = controller(frame_feats, temperature=1.0)

                mean_entropy = _results["entropies"].mean().item()
                epoch_loss += loss.item()

                actions = controller.inference(frame_feats, sel_nums=100).cpu().numpy()[0]

            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                f"Epoch {epoch + 1}, "
                f"loss: {epoch_loss / (epoch + 1)}, "
                f"entropy: {mean_entropy}, "
                f"actions: {actions[:30]}"
            )

            if mean_entropy <= 1.8:
                print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                    f"[INFO] Early stopping at epoch {epoch + 1} with entropy {mean_entropy}.")
                
                state = {
                    'epoch': epoch + 1,
                    'controller_state': controller.state_dict(),
                    'args': args,
                }
                
                fname = os.path.join(args.save_dir, f"{epoch + 1}_warm_up.pth")
                torch.save(state, fname)
                print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} "
                    f"[INFO] Save warm-up checkpoint to {fname}.")
                
                break


if __name__ == "__main__":
    args = parse_args()
    main(args)