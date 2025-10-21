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
from accelerate import Accelerator

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
def evaluate_pcd(args, selector: FrameSelector, reconstructor: SelectedFrameReconstructor):
    """
    Evaluates the frame selection and reconstruction pipeline on the 7Scenes dataset.
    This function is designed to be called during the training loop.
    """
    datasets_all = {
    "7scenes": SevenScenes(
        split="test",
        ROOT="./data/7scenes",
        resolution=resolution,
        num_seq=1,
        full_video=True,
        kf_every=200,
    ),  # 20),
    }

    accelerator = Accelerator()
    device = accelerator.device

    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
    from dust3r.utils.geometry import geotrf
    from copy import deepcopy

    
    print("\n[EVAL] Starting evaluation on 7Scenes...")
    #selector.eval() # Set selector to evaluation mode
    
    # 1. Load 7Scenes dataset
    try:
        dataset = SevenScenes(
            split='test',
            ROOT=args.eval_dataset_path,
            resolution=(518, 378),
            num_seq=1,
            full_video=True,
            kf_every=200
        )
    except FileNotFoundError:
        print(f"[ERROR] 7Scenes dataset not found at {args.eval_dataset_path}. Skipping evaluation.")
        return {}
    
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode="avg_dis", gt_scale=True)

    for name_data, dataset in datasets_all.items():
        save_path = osp.join(args.output_dir, name_data)
        os.makedirs(save_path, exist_ok=True)
        log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

        acc_all = 0
        acc_all_med = 0
        comp_all = 0
        comp_all_med = 0
        nc1_all = 0
        nc1_all_med = 0
        nc2_all = 0
        nc2_all_med = 0

        fps_all = []
        time_all = []

        with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
            for data_idx in tqdm(idxs):
                batch = default_collate([dataset[data_idx]])
                print(len(batch))
                ignore_keys = set(
                    [
                        "depthmap",
                        "dataset",
                        "label",
                        "instance",
                        "idx",
                        "true_shape",
                        "rng",
                    ]
                )
                for view in batch:
                    for name in view.keys():  # pseudo_focal
                        if name in ignore_keys:
                            continue
                        if isinstance(view[name], tuple) or isinstance(
                            view[name], list
                        ):
                            view[name] = [
                                x.to(device, non_blocking=True) for x in view[name]
                            ]
                        else:
                            view[name] = view[name].to(device, non_blocking=True)
                
                print("[INFO] Running full sequence inference")
                start = time.time()
                images = [batch[i]["img"] for i in range(len(batch))]
                _, _, full_preds = infer_sequence(images, reconstructor, pred_required=True)
                fps = len(batch) / (end - start)
                end = time.time()
                print(f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}, FPS: {fps:.2f}")
                fps_all.append(fps)
                time_all.append(end - start)

                # Evaluation
                print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                pp_full_preds = preprocess_vggt_predictions(full_preds)
                ggt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                    criterion.get_all_pts3d_t(batch, pp_full_preds)
                )
                pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                    monitoring["pred_scale"],
                    monitoring["gt_scale"],
                    monitoring["pred_shift_z"],
                    monitoring["gt_shift_z"],
                )

                in_camera1 = None
                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []
                conf_all = []

                for j, view in enumerate(batch):
                    if in_camera1 is None:
                        in_camera1 = view["camera_pose"][0].cpu()

                    image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                    mask = view["valid_mask"].cpu().numpy()[0]

                    # pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
                    pts = pred_pts[j].cpu().numpy()[0]
                    conf = preds[j]["conf"].cpu().data.numpy()[0]
                    # mask = mask & (conf > 1.8)

                    pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                    H, W = image.shape[:2]
                    cx = W // 2
                    cy = H // 2
                    l, t = cx - 112, cy - 112
                    r, b = cx + 112, cy + 112
                    image = image[t:b, l:r]
                    mask = mask[t:b, l:r]
                    pts = pts[t:b, l:r]
                    pts_gt = pts_gt[t:b, l:r]

                    #### Align predicted 3D points to the ground truth
                    pts[..., -1] += gt_shift_z.cpu().numpy().item()
                    pts = geotrf(in_camera1, pts)

                    pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                    pts_gt = geotrf(in_camera1, pts_gt)

                    images_all.append((image[None, ...] + 1.0) / 2.0)
                    pts_all.append(pts[None, ...])
                    pts_gt_all.append(pts_gt[None, ...])
                    masks_all.append(mask[None, ...])
                    conf_all.append(conf[None, ...])

            images_all = np.concatenate(images_all, axis=0)
            pts_all = np.concatenate(pts_all, axis=0)
            pts_gt_all = np.concatenate(pts_gt_all, axis=0)
            masks_all = np.concatenate(masks_all, axis=0)

            scene_id = view["label"][0].rsplit("/", 1)[0]

            if "DTU" in name_data:
                threshold = 100
            else:
                threshold = 0.1

            pts_all_masked = pts_all[masks_all > 0]
            pts_gt_all_masked = pts_gt_all[masks_all > 0]
            images_all_masked = images_all[masks_all > 0]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                pts_all_masked.reshape(-1, 3)
            )
            pcd.colors = o3d.utility.Vector3dVector(
                images_all_masked.reshape(-1, 3)
            )
            #o3d.io.write_point_cloud(
            #    os.path.join(
            #        save_path, f"{scene_id.replace('/', '_')}-mask.ply"
            #    ),
            #    pcd,
            #)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(
                pts_gt_all_masked.reshape(-1, 3)
            )
            pcd_gt.colors = o3d.utility.Vector3dVector(
                images_all_masked.reshape(-1, 3)
            )
            #o3d.io.write_point_cloud(
            #    os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
            #    pcd_gt,
            #)

            trans_init = np.eye(4)

            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            transformation = reg_p2p.transformation

            pcd = pcd.transform(transformation)
            pcd.estimate_normals()
            pcd_gt.estimate_normals()

            gt_normal = np.asarray(pcd_gt.normals)
            pred_normal = np.asarray(pcd.normals)

            acc, acc_med, nc1, nc1_med = accuracy(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            comp, comp_med, nc2, nc2_med = completion(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            print(
                f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
            )
            print(
                f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                file=open(log_file, "a"),
            )

            acc_all += acc
            comp_all += comp
            nc1_all += nc1
            nc2_all += nc2

            acc_all_med += acc_med
            comp_all_med += comp_med
            nc1_all_med += nc1_med
            nc2_all_med += nc2_med

            # release cuda memory
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    # Get depth from pcd and run TSDFusion
    if accelerator.is_main_process:
        to_write = ""
        # Copy the error log from each process to the main error log
        for i in range(8):
            if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                break
            with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                to_write += f_sub.read()

        with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
            log_data = to_write
            metrics = defaultdict(list)
            for line in log_data.strip().split("\n"):
                match = regex.match(line)
                if match:
                    data = match.groupdict()
                    # Exclude 'scene_id' from metrics as it's an identifier
                    for key, value in data.items():
                        if key != "scene_id":
                            metrics[key].append(float(value))
                    metrics["nc"].append(
                        (float(data["nc1"]) + float(data["nc2"])) / 2
                    )
                    metrics["nc_med"].append(
                        (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                    )
            mean_metrics = {
                metric: sum(values) / len(values)
                for metric, values in metrics.items()
            }

            c_name = "mean"
            print_str = f"{c_name.ljust(20)}: "
            for m_name in mean_metrics:
                print_num = np.mean(mean_metrics[m_name])
                print_str = print_str + f"{m_name}: {print_num:.3f} | "
            print_str = print_str + "\n"
            f.write(to_write + print_str)
    
            


def main(args):
    device=args.device
    reconstructor = SelectedFrameReconstructor(args).to(device)
    reconstructor.eval()
    selector = FrameSelector(args, feat_dim=args.feat_dim).to(device)
    selector.eval()
    
    evaluate_pcd(args, selector, reconstructor)



if __name__ == "__main__":
    from config import parse_args
    args = parse_args()
    main(args)