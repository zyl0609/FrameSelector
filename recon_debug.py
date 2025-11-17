
import sys
sys.path.append("./FastVGGT")
import os
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import viser

from models.frame_recon import SelectedFrameReconstructor
from models.controller import Controller
from config import parse_args
from models.data_utils import load_sample_frames

from typing import Optional

class VGGTViserViewer:
    def __init__(self,
                 pts: np.ndarray,
                 conf: np.ndarray,
                 img: np.ndarray,
                 port: int = 8080,
                 default_keep: float = 80.0):
        self.pts  = pts
        self.conf = conf
        self.img  = img
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")

        # 颜色归一化到 0~1
        if self.img.max() > 1.0:
            self.img = self.img.astype(np.float32) / 255.0
        self.img = np.clip(self.img, 0, 1)

        # ------------------ GUI ------------------
        self.keep_slider = self.server.add_gui_slider(
            "Keep %",
            min=0.0,
            max=100.0,
            step=0.1,
            initial_value=default_keep,
        )
        self.psize_slider = self.server.add_gui_slider(
            "Point size",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=0.003,
        )

        self.pc_handle: Optional[viser.PointCloudHandle] = None
        self._update_pc()          # 首次绘制

        # 监听滑条
        @self.keep_slider.on_update
        def _(_):
            self._update_pc()

        @self.psize_slider.on_update
        def _(_):
            if self.pc_handle is not None:
                self.pc_handle.point_size = self.psize_slider.value

    def _update_pc(self):
        keep_ratio = self.keep_slider.value / 100.0
        idx_sorted = np.argsort(self.conf)[::-1]          # 降序
        keep_num   = max(1, int(len(idx_sorted) * keep_ratio))
        idx_keep   = idx_sorted[:keep_num]

        pts_filt  = self.pts[idx_keep]
        color_filt = self.img[idx_keep]                  # 直接使用 image 颜色

        if self.pc_handle is not None:
            self.pc_handle.remove()
        self.pc_handle = self.server.add_point_cloud(
            name="/vggt_pts",
            points=pts_filt,
            colors=color_filt,
            point_size=self.psize_slider.value,
        )


"""
@torch.no_grad()
def main(args):
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seq_dir = "/opt/ml/code/cvpr2026/data/rgb_7scenes/pumpkin/seq-07"
    indices, frames = load_sample_frames(
        seq_dir, 
        pil_mode=True,
        max_frames=1000
    )

    recon = SelectedFrameReconstructor(args).to(args.device)
    recon.eval()

    selector = FrameSelector(args, 2048)

    args.ckpt_path = "./ckpt/pumpkin-seq-07/keep-0.1-20251030_004858/best.pth"
    if os.path.isfile(args.ckpt_path):
        print(f"[INFO] Load selector weight from {args.ckpt_path}")
        state = torch.load(args.ckpt_path, map_location='cpu')
        selector.load_state_dict(state["controller_state"], strict=True)
    selector = selector.to(args.device)
    selector.eval()

    # uniform sample
    keep_ratio = args.hard_ratio
    sample_step = round(1.0 / keep_ratio)

    full_preds, _ = recon(frames[::sample_step])
    recon.free_image_cache()

    for key in full_preds.keys():
        if isinstance(full_preds[key], torch.Tensor):
            full_preds[key] = full_preds[key].cpu().numpy()

    full_rgb = (full_preds["images"].transpose(0, 2, 3, 1).reshape(-1, 3)) # (N,3)
    full_xyz = full_preds["world_points"].reshape(-1, 3)          # (N,3)
    full_conf = full_preds["world_points_conf"].reshape(-1)       # (N,)

    #viewer = VGGTViserViewer(full_xyz, full_conf, full_rgb, port=args.port, default_keep=80)
    #print(f"[Viser] Open http://localhost:{args.port}")
    #try:
    #    while True:
    #        time.sleep(0.1)
    #except KeyboardInterrupt:
    #    print("Shutdown.")

    # Run full sequence to get embedding
    frame_feats = recon.get_frame_feat(frames)
    recon.free_image_cache()
    # Select frames 
    logits, _ = selector(frame_feats)
    k = max(1, round(keep_ratio * len(frames)))
    _, top_idx = torch.topk(logits.squeeze(), k=k, dim=0)
    keep_idx = sorted(top_idx.cpu().numpy().tolist())
    print(f"[INFO] Keep index: {keep_idx}")

    selected_images = [frames[i] for i in keep_idx]
    dropped_preds, _ = recon(selected_images)
    recon.free_image_cache()
    for key in dropped_preds.keys():
        if isinstance(dropped_preds[key], torch.Tensor):
            dropped_preds[key] = dropped_preds[key].cpu().numpy()
    
    dropped_rgb = (dropped_preds["images"].transpose(0, 2, 3, 1).reshape(-1, 3)) # (N,3)
    dropped_xyz = dropped_preds["world_points"].reshape(-1, 3)          # (N,3)
    dropped_conf = dropped_preds["world_points_conf"].reshape(-1)       # (N,)
    viewer = VGGTViserViewer(dropped_xyz, dropped_conf, dropped_rgb, port=args.port, default_keep=80)
    print(f"[Viser] Open http://localhost:{args.port}")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutdown.")
    plt.close()
"""


@torch.no_grad()
def main(args):
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seq_dir = "/home/lidabao/projects/code/cvpr2026/data/7scenes/pumpkin/seq-01"
    indices, frames = load_sample_frames(
        seq_dir, 
        pil_mode=True,
        max_frames=100
    )

    recon = SelectedFrameReconstructor(args).to(args.device)
    recon.eval()

    device = args.device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # uniform sample
    with torch.amp.autocast(device, enabled=True, dtype=dtype):
        full_preds, _ = recon(frames[::10])
    recon.free_image_cache()

    for key in full_preds.keys():
        if isinstance(full_preds[key], torch.Tensor):
            full_preds[key] = full_preds[key].float().cpu().numpy()

    full_rgb = (full_preds["images"].transpose(0, 2, 3, 1).reshape(-1, 3)) # (N,3)
    full_xyz = full_preds["world_points"].reshape(-1, 3)          # (N,3)
    full_conf = full_preds["world_points_conf"].reshape(-1)       # (N,)

    viewer = VGGTViserViewer(full_xyz, full_conf, full_rgb, port=args.port, default_keep=80)
    print(f"[Viser] Open http://localhost:{args.port}")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutdown.")
    


if __name__ == "__main__":
    args = parse_args()
    main(args)