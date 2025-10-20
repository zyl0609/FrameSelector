import os
import sys
import numpy as np
import torch
import torch.nn as nn
import open3d

sys.path.append("./vggt")

import PIL
from PIL import Image
from typing import List, Dict
from pathlib import Path

class SelectedFrameReconstructor(nn.Module):
    def __init__(self, args):
        """
        Reconstructor using selected frames and a pre-trained teacher model to generate predictions (e.g. depth maps, point clouds).

        :param args: configurations from config.py, including atributes `teacher_name`, `device` etc.

        """
        super().__init__()
        self.teacher_name = args.teacher_name
        self.device = args.device
        
        self.args = args # for other methods to use

        self._load_teacher()

    
    def forward(self, sample_frames: List):
        if self.teacher_name == "vggt":
            pred_dict = self._vggt_inference(sample_frames)

        else:
            raise ValueError(f"Only support VGGT now")
        
        return pred_dict


    def _project_world_points_to_images(
        self,
        images: torch.Tensor,
        world_points: torch.Tensor,
        world_points_conf: torch.Tensor,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor
    ):
        """
        Project 3D world points to 2D image plane using camera parameters.
        All operations are batched and executed in a single CUDA kernel launch
        to handle large frame counts (S=300~500).

        :param images: (S, 3, H, W) tensor of input RGB images.
        :param world_points: (S, H, W, 3) tensor of 3D points in world coordinates.
        :param world_points_conf: (S, H, W) tensor of confidence scores for each point.
        :param extrinsic: (S, 3, 4) tensor of camera extrinsic matrices (world-to-camera).
        :param intrinsic: (S, 3, 3) tensor of camera intrinsic matrices.

        :return rgb_map: (S, 3, H, W) tensor of rendered RGB images.
        :return depth_map: (S, 1, H, W) tensor of rendered depth maps.
        :return conf_map: (S, 1, H, W) tensor of rendered confidence maps.
        :return mask_map: (S, 1, H, W) tensor of valid pixel masks.
        """
        S, C, H, W = images.shape
        device = images.device
        dtype  = images.dtype
        N = S * H * W                      # total number of points

        # flatten tensors
        pts3d = world_points.view(N, 3)                    # (N, 3)
        conf  = world_points_conf.view(N)                  # (N,)
        rgb   = images.permute(0, 2, 3, 1).reshape(N, 3)   # (N, 3)

        # global pixel index
        b_idx = torch.arange(S, device=device).view(S, 1, 1).expand(S, H, W).reshape(N)
        v_idx = torch.arange(H, device=device).view(1, H, 1).expand(S, H, W).reshape(N)
        u_idx = torch.arange(W, device=device).view(1, 1, W).expand(S, H, W).reshape(N)
        pix_idx = b_idx * H * W + v_idx * W + u_idx        # (N,)

        # world → camera
        pts_homo = torch.cat([pts3d, torch.ones(N, 1, device=device, dtype=dtype)], dim=1)  # (N, 4)
        E_b = extrinsic.to(dtype)[b_idx]                        # (N, 3, 4)
        cam_pts = torch.bmm(E_b, pts_homo.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        z = cam_pts[:, 2]                                       # (N,)

        # camera → pixel
        K_b = intrinsic.to(dtype)[b_idx]                        # (N, 3, 3)
        uv_homo = torch.bmm(K_b, cam_pts.unsqueeze(-1)).squeeze(-1)   # (N, 3)
        uv = uv_homo[:, :2] / (uv_homo[:, 2:] + 1e-7)            # (N, 2)
        u, v = uv[:, 0], uv[:, 1]

        # valid mask
        valid_mask = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)  # (N,)
        #conf_mask = conf >= 0.45
        #mask &= conf_mask
        u = u.clamp(0, W - 1)
        v = v.clamp(0, H - 1)

        # update pixel index after clamp
        pix_idx = b_idx * H * W + v.long() * W + u.long()

        # bilinear weights
        u0 = u.long(); u1 = torch.min(u0 + 1, torch.tensor(W - 1, device=device))
        v0 = v.long(); v1 = torch.min(v0 + 1, torch.tensor(H - 1, device=device))
        du = (u - u0.float()).to(dtype)
        dv = (v - v0.float()).to(dtype)
        w00 = ((1 - du) * (1 - dv)).to(dtype) * valid_mask.to(dtype)
        w01 = ((1 - du) * dv).to(dtype)     * valid_mask.to(dtype)
        w10 = (du * (1 - dv)).to(dtype)     * valid_mask.to(dtype)
        w11 = (du * dv).to(dtype)         * valid_mask.to(dtype)

        pix00 = b_idx * H * W + v0 * W + u0
        pix01 = b_idx * H * W + v1 * W + u0
        pix10 = b_idx * H * W + v0 * W + u1
        pix11 = b_idx * H * W + v1 * W + u1

        # depth buffer (z-buffer)
        depth_buf = torch.full((N,), float('inf'), device=device, dtype=torch.float32)
        depth_buf.scatter_reduce_(0, pix_idx, z, reduce='amin', include_self=True)

        # hit mask (only the closest)
        hit = (z == depth_buf[pix_idx]) & valid_mask

        # color & confidence accumulation
        rgb_buf  = torch.zeros((N, 3), device=device, dtype=dtype)
        conf_buf = torch.zeros((N,),   device=device, dtype=dtype)

        # accumulate bilinear contributions
        for w, pix in [(w00, pix00), (w01, pix01), (w10, pix10), (w11, pix11)]:
            w_hit = w * hit.to(dtype)
            rgb_buf.index_add_(0, pix, (w_hit.unsqueeze(-1) * rgb).to(dtype))
            conf_buf.index_add_(0, pix, (w_hit * conf).to(dtype))
        
        # to avoid over-exposure pixel
        over_exposed_mask = (rgb_buf > 1.0).any(dim=1)        # (N,)
        rgb_buf = rgb_buf.clamp(0, 1)

        # reshape to image layout
        depth_map = depth_buf.view(S, H, W, 1).permute(0, 3, 1, 2)  # (S, 1, H, W)
        rgb_map   = rgb_buf.view(S, H, W, 3).permute(0, 3, 1, 2)    # (S, 3, H, W)
        conf_map  = conf_buf.view(S, H, W, 1).permute(0, 3, 1, 2)   # (S, 1, H, W)

        mask_map  = ((depth_buf < float('inf')) & (~over_exposed_mask)).view(S, 1, H, W).float()

        return rgb_map, depth_map, conf_map, mask_map
        
    
    def _load_teacher(self):
        """
        Load the pre-trained teacher model.
        """
        if self.teacher_name == 'vggt':
            
            from vggt.models.vggt import VGGT
            self.model = VGGT()
            ckpt = torch.load(self.args.vggt_ckpt, map_location='cpu')
            self.model.load_state_dict(ckpt)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.infer_fn = self._vggt_inference
            self.use_point_map = self.args.vggt_use_point_map
            print(f"[INFO] Loaded VGGT model from {os.path.abspath(self.args.vggt_ckpt)}")

        elif self.teacher_name == 'cut3r':
            # TODO: later
            raise NotImplementedError("CUT3R teacher coming soon")
        
        elif self.teacher_name == 'dust3r':
            raise NotImplementedError("Dust3r teacher coming soon")
        
        else:
            raise ValueError(f"unknown teacher {self.teacher_name}")
        

    @torch.no_grad()
    def _vggt_inference(self, sample_frames: List) -> Dict:
        """
        Inference using VGGT model.

        :param images: A list of PIL.Image or a list of preprocessed torch.Tensor.
        :return: A torch.Tensor dictionary containing:
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        from data_utils import load_and_preprocess_sample_frames
        from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        if isinstance(sample_frames[0], PIL.Image.Image):
            # Training: PIL.Image.Image List
            images = load_and_preprocess_sample_frames(sample_frames, target_size=self.args.vggt_imgsz).to(self.device)
        elif isinstance(sample_frames[0], torch.Tensor):
            # Validation/Test: preprocessed Tensor List
            images = torch.stack(sample_frames).to(self.device)
        else:
            raise TypeError(f"Unsupported input type for sample_frames: {type(sample_frames[0])}")


        with torch.cuda.amp.autocast(dtype=dtype):
            predictions, aggregated_token_embedding = self.model(images)    # dictionary

        # get pose and intrinsics
        extrinsics_cam, intrinsics_cam = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
        predictions['extrinsic'] = extrinsics_cam
        predictions['intrinsic'] = intrinsics_cam

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].squeeze(0)

        #cam_to_world_mat = closed_form_inverse_se3(predictions['extrinsic'])  # (S, 4, 4)
        #cam_to_world = cam_to_world_mat[:, :3, :]  # (S, 3, 4)
        
        if self.use_point_map:
            world_points = predictions['world_points']  # (S, H, W, 3)
            conf_map = predictions['world_points_conf']  # (S, H, W)
        else:
            # compute world points from depth map
            depth_map = predictions['depth']  # (S, H, W, 1)
            depth_conf = predictions['depth_conf']  # (S, H, W)

            world_points = unproject_depth_map_to_point_map(
                depth_map, predictions['extrinsic'], predictions['intrinsic']) # (S, H, W, 3)
            # use depth confidence as point confidence
            predictions['world_points'] = torch.from_numpy(world_points).to(dtype=dtype, device=images.device)

        return predictions, aggregated_token_embedding