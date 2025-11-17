import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import PIL
from PIL import Image
from typing import List, Dict, Union
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
        self._image_cache = None

    
    def forward(
        self, 
        sample_frames: List[Union[Image.Image, torch.Tensor]],
        squeeze_batch_dim: bool = True
    ):
        if self.teacher_name in ["vggt", "m_vggt", "fast_vggt"]:
            pred_dict, frame_feat = self.infer_fn(sample_frames, squeeze_batch_dim)
            return pred_dict, frame_feat
        else:
            raise ValueError(f"[ERROR] Only support VGGT now")
        

    def get_frame_feat(self, sample_frames: List[Union[Image.Image, torch.Tensor]]):
        if self.teacher_name in ["vggt", 'm_vggt', 'fast_vggt']:
            embeddings = self._get_vggt_embedding(sample_frames)

        else:
            raise ValueError(f"[ERROR] Only support VGGT now")
            
        return embeddings
    
    
    def free_image_cache(self):
        """Free cache manually."""
        self._image_cache = None
        torch.cuda.empty_cache()
        
        
    def _prerocess_images(self, sample_frames):
        if self._image_cache is None:
            #print(f"[INFO] Preprocess {len(sample_frames)} images to {self.teacher_name} format.")
            if self.teacher_name in ['vggt', 'm_vggt', 'fast_vggt']:
                if isinstance(sample_frames, List):
                    if isinstance(sample_frames[0], Image.Image):
                        from utility.data_utils import load_and_preprocess_sample_frames
                        self._image_cache = load_and_preprocess_sample_frames(sample_frames, target_size=self.args.vggt_imgsz).to(self.device)
                    elif isinstance(sample_frames[0], torch.Tensor):
                        self._image_cache = torch.cat(sample_frames, dim=0).to(self.device)    
                    else:
                        raise TypeError(f"Unsupported input type for sample_frames: {type(sample_frames[0])}")
                elif isinstance(sample_frames, torch.Tensor):
                    self._image_cache = sample_frames.to(self.device) # already torch.Tensor
                
                else:
                    raise TypeError(f"Unsupported input type for sample_frames: {type(sample_frames[0])}")
            else:
                raise TypeError(f"Unsupported model type: {self.teacher_name}")

        return self._image_cache


    # GPU consumption is large
    def _batch_project_world_points_to_images(
        self,
        images: torch.Tensor,
        world_points: torch.Tensor,
        world_points_conf: torch.Tensor,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        conf_threshold: float=0.5
    ):
        """
        Project 3D world points to 2D image plane using camera parameters.
        All operations are batched and executed in a single CUDA kernel launch
        to handle large frame counts (S=300~500).

        :param images: (K, 3, H, W) tensor of input RGB images.
        :param world_points: (K, H, W, 3) tensor of 3D points in world coordinates.
        :param world_points_conf: (K, H, W) tensor of confidence scores for each point.
        :param extrinsic: (S, 3, 4) tensor of camera extrinsic matrices (world-to-camera).
        :param intrinsic: (S, 3, 3) tensor of camera intrinsic matrices.

        :return rgb_map: (S, 3, H, W) tensor of rendered RGB images.
        :return depth_map: (S, 1, H, W) tensor of rendered depth maps.
        :return conf_map: (S, 1, H, W) tensor of rendered confidence maps.
        :return mask_map: (S, 1, H, W) tensor of valid pixel masks.
        """
        K, C, H, W = images.shape
        S = extrinsic.shape[0]
        device = images.device
        dtype  = images.dtype

        # flatten
        pts3d = world_points.reshape(-1, 3)                       # (N, 3)  N=K*H*W
        conf  = world_points_conf.reshape(-1)                     # (N,)
        rgb   = images.permute(0, 2, 3, 1).reshape(-1, 3)         # (N, 3)

        # construct index of source frames
        k_idx = torch.arange(K, device=device).view(K, 1, 1).expand(K, H, W).reshape(-1)  # (N,)

        # copy S times
        N = pts3d.shape[0]
        pts3d = pts3d.unsqueeze(1).expand(N, S, 3).reshape(N * S, 3)
        conf  = conf.unsqueeze(1).expand(N, S).reshape(N * S)
        rgb   = rgb.unsqueeze(1).expand(N, S, 3).reshape(N * S, 3)
        k_idx = k_idx.unsqueeze(1).expand(N, S).reshape(N * S)
        s_idx = torch.arange(S, device=device).view(1, S).expand(N, S).reshape(N * S)
        M = N * S

        # world → camera
        pts_homo = torch.cat([pts3d, torch.ones(M, 1, device=device, dtype=dtype)], dim=1)  # (M,4)
        E = extrinsic[s_idx].to(dtype)                           # (M,3,4)
        cam_pts = torch.bmm(E, pts_homo.unsqueeze(-1)).squeeze(-1)  # (M,3)
        z = cam_pts[:, 2]

        # camera → pixel
        Kmat = intrinsic[s_idx].to(dtype)                        # (M,3,3)
        uv_homo = torch.bmm(Kmat, cam_pts.unsqueeze(-1)).squeeze(-1)
        uv = uv_homo[:, :2] / (uv_homo[:, 2:] + 1e-7)
        u, v = uv[:, 0], uv[:, 1]

        valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (conf >= conf_threshold)
        u = u.clamp(0, W - 1)
        v = v.clamp(0, H - 1)

        # destination pixel indices
        pix_idx = s_idx * H * W + v.long() * W + u.long()  # (M,)

        # construct frame-independent Z-buffer
        depth_buf = torch.full((S * H * W,), float('inf'), device=device, dtype=torch.float32)
        depth_buf.scatter_reduce_(0, pix_idx, z, reduce='amin', include_self=True)
        hit = (z == depth_buf[pix_idx]) & valid  # (M,)

        # bilinear interpolation
        u0 = u.long(); u1 = torch.min(u0 + 1, torch.tensor(W - 1, device=device))
        v0 = v.long(); v1 = torch.min(v0 + 1, torch.tensor(H - 1, device=device))
        du = (u - u0.float()).to(dtype)
        dv = (v - v0.float()).to(dtype)
        w00 = ((1 - du) * (1 - dv)).to(dtype) * hit.to(dtype)
        w01 = ((1 - du) * dv).to(dtype)     * hit.to(dtype)
        w10 = (du * (1 - dv)).to(dtype)     * hit.to(dtype)
        w11 = (du * dv).to(dtype)         * hit.to(dtype)

        pix00 = s_idx * H * W + v0 * W + u0
        pix01 = s_idx * H * W + v1 * W + u0
        pix10 = s_idx * H * W + v0 * W + u1
        pix11 = s_idx * H * W + v1 * W + u1
        rgb_buf  = torch.zeros((S * H * W, 3), device=device, dtype=dtype)
        conf_buf = torch.zeros((S * H * W,),   device=device, dtype=dtype)

        for w, pix in [(w00, pix00), (w01, pix01), (w10, pix10), (w11, pix11)]:
            rgb_buf.index_add_(0, pix, (w.unsqueeze(-1) * rgb).to(dtype))
            conf_buf.index_add_(0, pix, w * conf)

        # post-process
        over_exposed = (rgb_buf > 1.0).any(dim=1)
        rgb_buf = rgb_buf.clamp(0, 1)

        rgb_map   = rgb_buf.view(S, H, W, 3).permute(0, 3, 1, 2)
        depth_map = depth_buf.view(S, H, W, 1).permute(0, 3, 1, 2)
        conf_map  = conf_buf.view(S, H, W, 1).permute(0, 3, 1, 2)
        mask_map  = ((depth_buf < float('inf')) & (~over_exposed)).view(S, H, W, 1).float()

        return rgb_map, depth_map, conf_map, mask_map


    def project_world_points_to_images(
        self,
        images: torch.Tensor,
        world_points: torch.Tensor,
        world_points_conf: torch.Tensor,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        conf_threshold: float=0.75
    ):
        """
        Project 3D world points to 2D image plane using camera parameters.
        All operations are batched and executed in a single CUDA kernel launch
        to handle large frame counts (S=300~500).

        :param images: (K, 3, H, W) tensor of input RGB images.
        :param world_points: (K, H, W, 3) tensor of 3D points in world coordinates.
        :param world_points_conf: (K, H, W) tensor of confidence scores for each point.
        :param extrinsic: (S, 3, 4) tensor of camera extrinsic matrices (world-to-camera).
        :param intrinsic: (S, 3, 3) tensor of camera intrinsic matrices.

        :return rgb_map: (S, 3, H, W) tensor of rendered RGB images.
        :return depth_map: (S, 1, H, W) tensor of rendered depth maps.
        :return conf_map: (S, 1, H, W) tensor of rendered confidence maps.
        :return mask_map: (S, 1, H, W) tensor of valid pixel masks.
        """
        K, _, H, W = images.shape
        S = extrinsic.shape[0]
        device = images.device
        dtype = images.dtype

        rgb_out   = torch.zeros((S, 3, H, W), device=device, dtype=dtype)
        depth_out = torch.full((S, 1, H, W), float('inf'), device=device, dtype=torch.float32)
        conf_out  = torch.zeros((S, 1, H, W), device=device, dtype=dtype)
        mask_out  = torch.zeros((S, 1, H, W), device=device, dtype=dtype)

        # flatten
        pts3d = world_points.reshape(-1, 3)                       # (N, 3)  N=K*H*W
        conf  = F.sigmoid(world_points_conf.reshape(-1))          # (N,)
        rgb   = images.permute(0, 2, 3, 1).reshape(-1, 3)         # (N, 3)

        for s in range(S):
            E = extrinsic[s].to(dtype)      # (3,4)
            Kmat = intrinsic[s].to(dtype)   # (3,3)

            # world → camera
            pts_homo = torch.cat([pts3d, torch.ones_like(pts3d[:, :1])], dim=1)  # (N,4)
            cam_pts = (E @ pts_homo.T).T                                    # (N,3)
            z = cam_pts[:, 2]

            # camera → pixel
            uv_homo = (Kmat @ cam_pts.T).T                                  # (N,3)
            uv = uv_homo[:, :2] / (uv_homo[:, 2:] + 1e-7)
            u, v = uv[:, 0], uv[:, 1]

            valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (conf >= conf_threshold)
            u = u.clamp(0, W-1)
            v = v.clamp(0, H-1)

            # select the nearest depth
            pix_idx = v.long() * W + u.long()                              # (N,)
            depth_buf = torch.full((H*W,), float('inf'), device=device, dtype=torch.float32)
            depth_buf.scatter_reduce_(0, pix_idx[valid], z[valid], reduce='amin', include_self=True)
            hit = (z <= depth_buf[pix_idx] + 1e-3) & valid

            # bilinear interpolation
            u0 = u.long(); u1 = torch.min(u0+1, torch.tensor(W-1, device=device))
            v0 = v.long(); v1 = torch.min(v0+1, torch.tensor(H-1, device=device))
            du = (u - u0.float()).to(dtype)
            dv = (v - v0.float()).to(dtype)

            w00 = ((1-du)*(1-dv))*hit.to(dtype)
            w01 = ((1-du)*dv)    *hit.to(dtype)
            w10 = (du*(1-dv))    *hit.to(dtype)
            w11 = (du*dv)        *hit.to(dtype)

            pix00 = v0*W + u0
            pix01 = v1*W + u0
            pix10 = v0*W + u1
            pix11 = v1*W + u1

            rgb_buf  = torch.zeros((H*W, 3), device=device, dtype=dtype)
            conf_buf = torch.zeros((H*W,),   device=device, dtype=dtype)
            wsum_buf = torch.zeros((H*W,),   device=device, dtype=dtype)

            for w, pix in [(w00, pix00), (w01, pix01), (w10, pix10), (w11, pix11)]:
                rgb_buf.index_add_(0, pix, (w.unsqueeze(-1)*rgb).to(dtype))
                conf_buf.index_add_(0, pix, w*conf)
                wsum_buf.index_add_(0, pix, w)
            
            valid_pixel = wsum_buf > 1e-3
            rgb_buf[valid_pixel] /= wsum_buf[valid_pixel].unsqueeze(-1)

            over_exposed = (rgb_buf > 1.0).any(dim=1)
            rgb_buf.clamp_(0, 1)

            rgb_out[s]   = rgb_buf.view(H, W, 3).permute(2, 0, 1)
            depth_out[s] = depth_buf.view(1, H, W)
            conf_out[s]  = conf_buf.view(1, H, W)
            mask_out[s]  = ((depth_buf < float('inf')) & (~over_exposed)).view(1, H, W).float()

            del depth_buf, rgb_buf, conf_buf

        return rgb_out, depth_out.permute(0, 2, 3, 1), \
            conf_out.permute(0, 2, 3, 1), mask_out
        
    
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

        elif self.teacher_name == 'm_vggt':
            from vggt.models.vggt import VGGT
            ckpt_path = self.args.m_vggt_ckpt
            self.model = VGGT(merging=None, enable_point=True)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            incompat = self.model.load_state_dict(ckpt, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.infer_fn = self._modified_vggt_inference
            self.use_point_map = False
            print(f"[INFO] Loaded modified VGGT model from {os.path.abspath(ckpt_path)}")

        elif self.teacher_name == 'fast_vggt':
            from vggt.models.vggt import VGGT
            ckpt_path = self.args.m_vggt_ckpt
            self.model = VGGT(merging=0, vis_attn_map=False, enable_point=True)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            incompat = self.model.load_state_dict(ckpt, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.infer_fn = self._modified_vggt_inference
            self.use_point_map = False
            print(f"[INFO] Loaded FastVGGT model from {os.path.abspath(ckpt_path)}")
        
        elif self.teacher_name == 'cut3r':
            # TODO: later
            raise NotImplementedError("CUT3R teacher coming soon")
        
        elif self.teacher_name == 'dust3r':
            raise NotImplementedError("Dust3r teacher coming soon")
        
        else:
            raise ValueError(f"unknown teacher {self.teacher_name}")
    
    
    @torch.no_grad()
    def _get_vggt_embedding(self, sample_frames: List[Union[Image.Image, torch.Tensor]]) -> torch.Tensor:
        """ 
        Get the VGGT embeddings from its aggregator.
        :param sample_frames: A list of PIL images or torch.Tensors
        """
        images = self._prerocess_images(sample_frames)
            
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
        last_feat = aggregated_tokens_list[-1]              # (B, S, L, 2C)
        patch_feat = last_feat[:, :, patch_start_idx:, :]   # (B, S, HW, 2C)
        frame_feat = patch_feat.mean(dim=2)                 # (B, S, 2C)
        return frame_feat
    

    @torch.no_grad()
    def _vggt_inference(self, sample_frames: List[Union[Image.Image, torch.Tensor]]) -> Dict:
        """
        Inference using VGGT model.

        :param images: A list of PIL.Image or a list of preprocessed torch.Tensor.
        :return predictions: A torch.Tensor dictionary containing:
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        :param embeddings: VGGT encoder's output. 
        """
        
        from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        images = self._prerocess_images(sample_frames)
        
        predictions, embeddings = self.model(images)    # dictionary

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
            # conf_map = predictions['world_points_conf']  # (S, H, W)
        else:
            # compute world points from depth map
            depth_map = predictions['depth']  # (S, H, W, 1)
            depth_conf = predictions['depth_conf']  # (S, H, W)

            world_points = unproject_depth_map_to_point_map(
                depth_map, predictions['extrinsic'], predictions['intrinsic']) # (S, H, W, 3)
            # use depth confidence as point confidence
            predictions['world_points'] = torch.from_numpy(world_points).float().to(device=images.device)
            predictions['world_points_conf'] = depth_conf # logits

        # convert logitis to probs
        predictions['world_points_conf'] = F.sigmoid(predictions['world_points_conf'])

        return predictions, embeddings
    

    @torch.no_grad()
    def _modified_vggt_inference(
        self, 
        sample_frames: List[Union[Image.Image, torch.Tensor]],
        squeeze_batch_dim: bool = True
    ) -> Dict:
        """
        Inference using VGGT model.

        :param images: A list of PIL.Image or a list of preprocessed torch.Tensor.
        :return predictions: A torch.Tensor dictionary containing:
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        :param embeddings: VGGT encoder's output. 
        """
        from vggt.utils.geometry import closed_form_inverse_se3
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = self._prerocess_images(sample_frames)
        predictions, frame_feats = self.model(images)    # dictionary

        # get pose and intrinsics
        extrinsics_cam, intrinsics_cam = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
        predictions['extrinsic'] = extrinsics_cam
        predictions['intrinsic'] = intrinsics_cam

        cam_to_world_mat = closed_form_inverse_se3(predictions['extrinsic'].squeeze(0))  # (S, 4, 4)
        cam_to_world = cam_to_world_mat[:, :3, :]  
        predictions["cam_to_world"] = cam_to_world.unsqueeze(0) # (S, 3, 4)

        if squeeze_batch_dim:
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].squeeze(0)

        ## Not support bfloat16
        #if not self.use_point_map:
        #    # only support depth unproject
        #    # compute world points from depth map
        #    depth_map = predictions['depth']  # (S, H, W, 1)
        #    depth_conf = predictions['depth_conf']  # (S, H, W)

        #    world_points = unproject_depth_map_to_point_map(
        #        depth_map, predictions['extrinsic'], predictions['intrinsic']) # (S, H, W, 3)
        #    # use depth confidence as point confidence
        #    predictions['world_points'] = torch.from_numpy(
        #        world_points.astype(np.float32)).to(device=images.device)
        #    predictions['world_points_conf'] = depth_conf # logits

        # convert logitis to probs
        #predictions['world_points_conf'] = F.sigmoid(predictions['world_points_conf'])

        return predictions, frame_feats