import time
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from PIL import Image
import open3d as o3d
import os.path as osp
from collections import defaultdict
from torch.utils.data._utils.collate import default_collate

from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


sys.path.append("./src")
try:
    import dust3r.heads
    import dust3r.utils.camera
    import dust3r.utils.geometry
    import dust3r.post_process
    #import dust3r.cloud_opt.commons
except ImportError as e:
    print(f"Warning: Pre-loading dust3r modules failed. This might cause issues. Error: {e}")

from controller import FrameSelector
from frame_recon import SelectedFrameReconstructor, infer_sequence
from eval_utils import preprocess_vggt_predictions

# Import evaluation utilities from CUT3R
from eval.mv_recon.data import SevenScenes
from eval.mv_recon.utils import accuracy, completion


@torch.no_grad()
def evaluate_pcd(
    args, 
    reconstructor: SelectedFrameReconstructor, 
    selector: FrameSelector=None, 
    val_epoch:int=None
):
    # Loading dataset
    from eval.mv_recon.data import SevenScenes
    dataset = SevenScenes(
        split='test',
        ROOT=args.eval_dataset_path,
        resolution=(518, 378),
        num_seq=1,
        full_video=True,
        kf_every=200
    )

    device = args.device

    # Points cloud evaluation criterion
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
    from dust3r.utils.geometry import geotrf
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode="avg_dis", gt_scale=False)

    # The log file path
    save_path = osp.join(args.eval_output_dir, "7scenes")
    os.makedirs(save_path, exist_ok=True)
    
    log_txt_name = "log_single_proc.txt"
    if not val_epoch:
        log_txt_name = f"log_val_{val_epoch}.txt"
        if selector is not None:
            log_txt_name = f"log_val_sel_{val_epoch}.txt"
    log_file = osp.join(save_path, log_txt_name)

    acc_all, acc_all_med = 0, 0
    comp_all, comp_all_med = 0, 0
    nc1_all, nc1_all_med = 0, 0
    nc2_all, nc2_all_med = 0, 0
    fps_all, time_all = [], []

    # Run sequentially
    for data_idx in tqdm(range(len(dataset)), desc="7Scenes-eval"):
        batch = default_collate([dataset[data_idx]]) # list of dict

        ignore_keys = {"depthmap", "dataset", "label", "instance", "idx",
                       "true_shape", "rng"}
        for view in batch:
            for k in list(view.keys()):
                if k in ignore_keys:
                    continue
                if isinstance(view[k], (list, tuple)):
                    view[k] = [v.to(device, non_blocking=True) for v in view[k]]
                else:
                    view[k] = view[k].to(device, non_blocking=True)

        # Reconstruction
        start = time.time()
        images = [batch[i]["img"] for i in range(len(batch))]
        if selector:
            embeddings = reconstructor.get_embedding(images)
            reconstructor.free_image_cache() # free images
            logits, _ = selector(embeddings)
            mask, _, _ = selector.sample(logits, temp=args.temperature, hard=True)
            keep_idx = torch.where(mask.squeeze() > 0.5)[0].cpu().numpy()
            assert keep_idx.size > 0
                
            images = [images[i] for i in keep_idx]
            batch = [batch[i] for i in keep_idx]
        
        _, preds, _ = infer_sequence(images, reconstructor, pred_required=True)
        reconstructor.free_image_cache() # free images
        elapsed = time.time() - start
        fps = len(batch) / elapsed
        fps_all.append(fps)
        time_all.append(elapsed)

        # Preprocess to Dust3R format
        pp_preds = preprocess_vggt_predictions(preds)
        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = \
            criterion.get_all_pts3d_t(batch, pp_preds)
        pred_scale, gt_scale, pred_shift_z, gt_shift_z = \
            monitoring["pred_scale"], monitoring["gt_scale"], \
            monitoring["pred_shift_z"], monitoring["gt_shift_z"]

        pts_all, pts_gt_all, masks_all, images_all = [], [], [], []
        in_camera1 = None
        for j, view in enumerate(batch):
            if in_camera1 is None:
                in_camera1 = view["camera_pose"][0].cpu()

            img = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]          # [H,W,3]
            msk = view["valid_mask"].cpu().numpy()[0]                        # [H,W]
            pts = pred_pts[j].cpu().numpy()[0]                               # [H,W,3]
            pts_gt = gt_pts[j].detach().cpu().numpy()[0]

            H, W = img.shape[:2]
            cx, cy = W // 2, H // 2
            l, t = cx - 112, cy - 112
            r, b = cx + 112, cy + 112
            img = img[t:b, l:r]
            msk = msk[t:b, l:r]
            pts = pts[t:b, l:r]
            pts_gt = pts_gt[t:b, l:r]

            # align
            pts[..., -1] += gt_shift_z.cpu().numpy().item()
            pts = geotrf(in_camera1, pts)
            pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
            pts_gt = geotrf(in_camera1, pts_gt)

            images_all.append((img[None] + 1.0) / 2.0)
            pts_all.append(pts[None])
            pts_gt_all.append(pts_gt[None])
            masks_all.append(msk[None])

        images_all = np.concatenate(images_all, 0)
        pts_all = np.concatenate(pts_all, 0)
        pts_gt_all = np.concatenate(pts_gt_all, 0)
        masks_all = np.concatenate(masks_all, 0)

        scene_id = batch[0]["label"][0].rsplit("/", 1)[0]

        threshold = 100 if "DTU" in args.eval_dataset_path else 0.1

        # Maks point cloud
        pts_m = pts_all[masks_all > 0]
        pts_gt_m = pts_gt_all[masks_all > 0]
        imgs_m = images_all[masks_all > 0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_m.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(imgs_m.reshape(-1, 3))

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_m.reshape(-1, 3))
        pcd_gt.colors = o3d.utility.Vector3dVector(imgs_m.reshape(-1, 3))

        # ICP alignment
        reg = o3d.pipelines.registration.registration_icp(
            pcd, pcd_gt, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        pcd = pcd.transform(reg.transformation)
        pcd.estimate_normals()
        pcd_gt.estimate_normals()

        # metrics
        acc, acc_med, nc1, nc1_med = accuracy(
            pcd_gt.points, pcd.points,
            np.asarray(pcd_gt.normals), np.asarray(pcd.normals))
        comp, comp_med, nc2, nc2_med = completion(
            pcd_gt.points, pcd.points,
            np.asarray(pcd_gt.normals), np.asarray(pcd.normals))

        log_line = (f"Idx: {scene_id}, Acc: {acc:.4f}, Comp: {comp:.4f}, "
                    f"NC1: {nc1:.4f}, NC2: {nc2:.4f} - "
                    f"Acc_med: {acc_med:.4f}, Compc_med: {comp_med:.4f}, "
                    f"NC1c_med: {nc1_med:.4f}, NC2c_med: {nc2_med:.4f}")
        print(log_line)
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

        acc_all += acc; comp_all += comp; nc1_all += nc1; nc2_all += nc2
        acc_all_med += acc_med; comp_all_med += comp_med
        nc1_all_med += nc1_med; nc2_all_med += nc2_med

        torch.cuda.empty_cache()

    # compute average metrics
    N = len(dataset)
    mean_line = (f"mean: Acc: {acc_all/N:.3f} | Comp: {comp_all/N:.3f} | "
                 f"NC1: {nc1_all/N:.3f} | NC2: {nc2_all/N:.3f} | "
                 f"Acc_med: {acc_all_med/N:.3f} | Comp_med: {comp_all_med/N:.3f} | "
                 f"NC1_med: {nc1_all_med/N:.3f} | NC2_med: {nc2_all_med/N:.3f}\n")
    print(mean_line)
    with open(log_file, "a") as f:
        f.write(mean_line)
    
            


def main(args):
    device=args.device

    # init by VGGT's checkpoint
    reconstructor = SelectedFrameReconstructor(args).to(device)
    reconstructor.eval()

    selector = FrameSelector(args, 2048)
    if os.path.isfile(args.ckpt_path):
        print(f"[INFO] Load selector weight from {args.ckpt_path}")
        state = torch.load(args.ckpt_path, map_location='cpu')
        selector.load_state_dict(state["controller_state"], strict=True)
    selector = selector.to(device)
    selector.eval()
    
    evaluate_pcd(args, reconstructor, selector)



if __name__ == "__main__":
    from config import parse_args
    args = parse_args()
    main(args)