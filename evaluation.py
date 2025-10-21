import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from PIL import Image
import open3d as o3d

# Add CUT3R path
sys.path.append("./src")

from controller import FrameSelector
from frame_recon import SelectedFrameReconstructor, infer_sequence

# Import evaluation utilities from CUT3R
from eval.mv_recon.data import SevenScenesData
from eval.mv_recon.utils import accuracy, completion, f_score

@torch.no_grad()
def evaluate_on_7scenes(args, selector: FrameSelector, reconstructor: SelectedFrameReconstructor, device: torch.device):
    """
    Evaluates the frame selection and reconstruction pipeline on the 7Scenes dataset.
    This function is designed to be called during the training loop.
    """
    print("\n[EVAL] Starting evaluation on 7Scenes...")
    selector.eval() # Set selector to evaluation mode
    
    # 1. Load 7Scenes dataset
    try:
        dataset = SevenScenesData(
            root=args.eval_dataset_path,
            split='test',
            resolution=(480, 640),
            full_video=True
        )
    except FileNotFoundError:
        print(f"[EVAL-ERROR] 7Scenes dataset not found at {args.eval_dataset_path}. Skipping evaluation.")
        return {}

    all_metrics = {'acc': [], 'comp': [], 'fscore': []}

    for i in tqdm(range(len(dataset)), desc="7Scenes Eval", leave=False):
        sample = dataset[i]
        images_pil = [Image.fromarray(img.numpy()) for img in sample['rgbs']]
        gt_pcd_o3d = sample['pts_3d']

        # 2. Run inference to get embeddings and select frames
        _, embedding, _, _ = infer_sequence(images_pil, reconstructor, embedding_required=True)
        logits, _ = selector(embedding)
        
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()
        keep_idx = torch.where(mask.squeeze() > 0.5)[0].cpu().numpy()
        
        if len(keep_idx) == 0:
            keep_idx = [torch.topk(probs, k=1)[1].item()]
        
        selected_images = [images_pil[i] for i in keep_idx]

        # 3. Reconstruct point cloud from selected frames
        _, _, pred_pcd_np, pred_pcd_conf = infer_sequence(
            selected_images, reconstructor, pcd_required=True
        )
        
        if pred_pcd_np is None or pred_pcd_np.size == 0:
            print(f"[EVAL-WARN] Empty point cloud for sequence {i}. Skipping.")
            continue

        pred_pcd_np = pred_pcd_np.reshape(-1, 3)
        pred_pcd_conf = pred_pcd_conf.reshape(-1)
        conf_mask = pred_pcd_conf > 0.5 
        pred_pcd_np = pred_pcd_np[conf_mask]

        if pred_pcd_np.shape[0] < 100: # Not enough points to evaluate
            print(f"[EVAL-WARN] Not enough points ({pred_pcd_np.shape[0]}) after filtering for sequence {i}. Skipping.")
            continue

        # 4. Align and Compute metrics
        gt_pcd_np = np.asarray(gt_pcd_o3d.points)
        
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pred_pcd_np)

        # ICP alignment
        threshold = 0.1 # As in launch.py
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_pred, gt_pcd_o3d, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        )
        pcd_pred.transform(reg_p2p.transformation)

        # Use CUT3R's metric functions which expect torch tensors
        acc = accuracy(torch.from_numpy(np.asarray(pcd_pred.points)).float(), torch.from_numpy(gt_pcd_np).float())
        comp = completion(torch.from_numpy(np.asarray(pcd_pred.points)).float(), torch.from_numpy(gt_pcd_np).float())
        fscore = 2 * (acc * comp) / (acc + comp + 1e-8)

        all_metrics['acc'].append(acc.item())
        all_metrics['comp'].append(comp.item())
        all_metrics['fscore'].append(fscore.item())

    # Aggregate results
    eval_results = {
        "7scenes_acc": np.mean(all_metrics['acc']) if all_metrics['acc'] else 0,
        "7scenes_comp": np.mean(all_metrics['comp']) if all_metrics['comp'] else 0,
        "7scenes_fscore": np.mean(all_metrics['fscore']) if all_metrics['fscore'] else 0,
    }

    print(f"[EVAL] Finished. F-Score: {eval_results['7scenes_fscore']:.4f}, "
          f"Acc: {eval_results['7scenes_acc']:.4f}, Comp: {eval_results['7scenes_comp']:.4f}")
    
    selector.train() # Set selector back to training mode
    return eval_results