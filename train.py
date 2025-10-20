import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import os, glob, shutil
import random
from PIL import Image
from tqdm import tqdm

from typing import Dict, List, Union

from data_utils import load_sample_frames, read_image_sequences
from config import parse_args
from controller import FrameSelector
from frame_recon import SelectedFrameReconstructor

# [添加] 预加载 dust3r 模块以解决循环导入问题
sys.path.append("./src")
try:
    import dust3r.heads
    import dust3r.utils.camera
    import dust3r.utils.geometry
    import dust3r.post_process
    import dust3r.cloud_opt.commons
except ImportError as e:
    print(f"Warning: Pre-loading dust3r modules failed. This might cause issues. Error: {e}")

import open3d as o3d


def set_random_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def infer_sequence(
    image_seq: List[Image.Image],
    reconstructor: SelectedFrameReconstructor,
    embedding_required=False,
    pcd_required=False
):
    """
    Given a list of PIL images, infer the reconstructed results using the reconstructor.
    :param image_seq: List of PIL images.
    :param reconstructor: The SelectedFrameReconstructor model.
    :param embedding_required: Whether to return the frame embeddings.
    :return rgb_map:  The projection of point clouds to RGB map tensor.
    :return embedding: (Optional) The frame embeddings from the teacher model.
    :return world_points: (Optional) The reconstructed 3D world points as a numpy array.
    :return world_points_conf: (Optional) The confidence of the reconstructed 3D world points as a numpy array.
    """
    with torch.no_grad():
        pred_dict, embedding = reconstructor(image_seq)

    with torch.no_grad():
        rgb_map, depth_map, conf_map, mask_map = reconstructor._project_world_points_to_images(
            pred_dict["images"], 
            pred_dict["world_points"], 
            pred_dict["world_points_conf"], 
            pred_dict["extrinsic"], 
            pred_dict["intrinsic"])

    world_points = None
    world_points_conf = None
    if pcd_required:
        world_points = pred_dict["world_points"].cpu().numpy() # (S, H, W, 3)
        world_points_conf = pred_dict["world_points_conf"].cpu().numpy() # (S, H, W)
    
    # Clean up large intermediate tensors immediately
    del pred_dict
        
    if embedding_required:
        # No need for detach() or requires_grad_() here as we are in no_grad context
        return rgb_map, embedding, world_points, world_points_conf
    else:
        del embedding
        return rgb_map, None, world_points, world_points_conf


def compute_reward(drop_render, gt_render):
    return -F.l1_loss(drop_render, gt_render)


def compute_reward_neighbor(
    drop_render: torch.Tensor, 
    gt_render: torch.Tensor, 
    keep_idx: torch.Tensor, 
    window_size: int=2
):
    """
    Compute reward based on local neighborhood L1 loss.
    :param drop_render: (S, 3, H, W) Rendered images from selected frames.
    :param gt_render: (S, 3, H, W) Pseudo-ground truth RGB images.
    :param keep_idx: (K,) Indices of selected frames.
    :param window_size: int Size of the neighborhood window on each side.
    :return: Mean reward over selected frames based on neighborhood L1 loss.
    """
    rewards = []
    device = drop_render.device

    for i in keep_idx:
        start = max(0, i - window_size)
        end   = min(gt_render.shape[0], i + window_size + 1)
        idx   = torch.arange(start, end, device=device)

        drop_render = drop_render[idx]          # (w,3,H,W)
        gt_render   = gt_render[idx]

        rewards.append(-F.l1_loss(drop_render, gt_render))

    return torch.stack(rewards).mean()


def select_topk(logits, k=None, ratio=None, hard=True):
    """
    logits: (B, S)  0~1概率
    k: int          固定选k帧
    ratio: float    按比例选
    hard: bool      True->返回0/1 mask，False->返回soft topk权重
    return:
        mask: (B, S)  0/1 或 soft权重
        log_prob: (B,)  选中的logits之和（用于RL）
    """
    B, S = logits.shape
    device = logits.device

    if k is None and ratio is not None:
        k = max(1, int(ratio * S))
    else:
        k = k or max(1, int(0.1 * S))

    # 取top-k索引
    scores = torch.sigmoid(logits)
    _, top_idx = torch.topk(scores, k, dim=1)  # (B, k)

    if hard:
        # 0/1 mask
        mask = torch.zeros_like(logits)
        mask.scatter_(1, top_idx, 1.0)
        # 梯度用straight-through：forward=0/1，backward=scores
        mask = (mask - scores).detach() + scores
    else:
        # soft：只保留top-k权重，其余置0
        mask = torch.zeros_like(scores)
        mask.scatter_(1, top_idx, scores.gather(1, top_idx))
        # 归一化到总和=k（可选）
        mask = mask * k / (mask.sum(dim=1, keepdim=True) + 1e-6)

    # 简单log_prob：选中帧的logits之和
    log_prob = logits.gather(1, top_idx).sum(dim=1)
    return mask, log_prob


def train_controller(args, selector, optimizer, reward, baseline, log_prob, entropy):
    selector.train()
    if baseline is None:
        baseline = reward
    else:
        baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * reward
        baseline = baseline.clone().detach()

    adv = reward - baseline
    loss = -log_prob * adv - args.entropy_coeff * entropy
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()
    if args.controller_grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.controller_grad_clip)
    optimizer.step()

    info = {
        "loss": loss.item(),
        "reward": reward.item(),
        "baseline": baseline.item(),
        "adv": adv.item(),
        "entropy": entropy.item(),
        "lr": optimizer.param_groups[0]['lr'],
    }
    return baseline, info


def save_checkpoint(state, epoch, reward, save_dir, is_best=False):
    """
    Save the training state to a checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"ckp_ep{epoch:04d}_r{reward:.4f}.pth")
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


def evaluate(args, selector, reconstructor):
    """
    Evaluate the frame selector by comparing reconstruction quality.
    Uses CUT3R's official scale-invariant criterion for alignment.
    """
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21 

    print("\n===== Starting Evaluation =====")
    device = args.device
    selector.eval()
    reconstructor.eval()

    # 使用 CUT3R 的评估工具，它会自动处理尺度对齐
    # norm_mode=False 和 gt_scale=True 是 7-Scenes 评估的标准设置
    eval_criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    # 假设评估数据集为7-Scenes
    from eval.mv_recon.data import SevenScenes
    dataset = SevenScenes(
        split="test",
        ROOT="./data/7scenes", # 请确保路径正确
        resolution=(512, 384),
        num_seq=1,
        full_video=True,
        kf_every=10 # 评估时可以适当降低帧率
    )
    
    results = {}

    for i in range(len(dataset)):
        views = dataset[i]
        scene_id = views[0]['label'].rsplit('/', 1)[0]
        print(f"\n--- Evaluating scene: {scene_id} ---")

        frames = [v['img'] for v in views]
        
        # 1. 使用完整序列进行重建 (作为对比基线)
        _, _, full_world_points, full_world_points_conf = infer_sequence(frames, reconstructor, pcd_required=True)
        
        # [修改] 增加置信度过滤
        conf_mask_full = full_world_points_conf.reshape(-1) > args.conf_threshold
        full_points = full_world_points.reshape(-1, 3)[conf_mask_full]

        # 2. 生成Ground-Truth点云 (Metric Scale)
        gt_points_list = []
        for view in views:
            pose_inv = np.linalg.inv(view['camera_pose'])
            cam_pts = view['pts3d'][view['valid_mask']]
            world_pts = (pose_inv[:3, :3] @ cam_pts.T + pose_inv[:3, 3:4]).T
            gt_points_list.append(world_pts)

        gt_points = np.concatenate(gt_points_list, axis=0)
        
        # 为减少计算量，可以对GT点云进行降采样
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
        gt_pcd = gt_pcd.voxel_down_sample(voxel_size=0.02)
        gt_points = np.asarray(gt_pcd.points)

        # 3. 使用FrameSelector选择帧并重建
        _, embedding, _, _ = infer_sequence(frames, reconstructor, embedding_required=True)
        logits, _ = selector(embedding)
        mask, _ = select_topk(logits, ratio=args.select_ratio, hard=True)
        keep_idx = torch.where(mask.squeeze() > 0.5)[0].cpu().numpy()

        if keep_idx.size == 0:
            scores = torch.sigmoid(logits.squeeze())
            _, top_idx = torch.topk(scores, k=max(1, int(args.select_ratio * len(frames))))
            keep_idx = top_idx.cpu().numpy()
        
        sel_images = [frames[i] for i in keep_idx]
        _, _, sel_world_points, sel_world_points_conf = infer_sequence(sel_images, reconstructor, pcd_required=True)

        # [修改] 增加置信度过滤
        conf_mask_sel = sel_world_points_conf.reshape(-1) > args.conf_threshold
        sel_points = sel_world_points.reshape(-1, 3)[conf_mask_sel]

        # 4. 使用 CUT3R 的 criterion 计算评估指标 (包含自动对齐)
        # Full vs GT
        # criterion 需要 torch tensor, 且在特定 device 上
        full_points_t = torch.from_numpy(full_points).to(device)
        gt_points_t = torch.from_numpy(gt_points).to(device)
        acc_full, _, comp_full, _, _, _ = eval_criterion(full_points_t, gt_points_t)
        
        # Selected vs GT
        sel_points_t = torch.from_numpy(sel_points).to(device)
        acc_sel, _, comp_sel, _, _, _ = eval_criterion(sel_points_t, gt_points_t)

        print(f"  [Full Sequence]    Accuracy: {acc_full.item():.4f}, Completion: {comp_full.item():.4f}")
        print(f"  [Selected Frames]  Accuracy: {acc_sel.item():.4f}, Completion: {comp_sel.item():.4f} ({len(sel_images)}/{len(frames)} frames)")

        results[scene_id] = {
            'acc_full': acc_full.item(), 'comp_full': comp_full.item(),
            'acc_sel': acc_sel.item(), 'comp_sel': comp_sel.item(),
            'num_selected': len(sel_images), 'num_total': len(frames)
        }


    # 打印平均结果
    avg_acc_full = np.mean([res['acc_full'] for res in results.values()])
    avg_comp_full = np.mean([res['comp_full'] for res in results.values()])
    avg_acc_sel = np.mean([res['acc_sel'] for res in results.values()])
    avg_comp_sel = np.mean([res['comp_sel'] for res in results.values()])
    avg_ratio = np.mean([res['num_selected'] / res['num_total'] for res in results.values()])

    print("\n===== Evaluation Summary =====")
    print(f"  [Full Sequence]    Avg Accuracy: {avg_acc_full:.4f}, Avg Completion: {avg_comp_full:.4f}")
    print(f"  [Selected Frames]  Avg Accuracy: {avg_acc_sel:.4f}, Avg Completion: {avg_comp_sel:.4f}")
    print(f"  Average frame selection ratio: {avg_ratio:.2%}")


def main(args):
    # Random seed setting
    device = args.device
    cudnn.benchmark = True
    set_random_seed(args.seed)

    reconstructor = SelectedFrameReconstructor(args).to(device)
    reconstructor.eval()
    selector = FrameSelector(args, feat_dim=args.feat_dim).to(device)
    optimizer = torch.optim.SGD(selector.parameters(), lr=args.controller_lr, momentum=0.9)

    if args.mode == 'eval':
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            ckp = torch.load(args.resume, map_location=device)
            selector.load_state_dict(ckp['controller_state'])
        evaluate(args, selector, reconstructor)
        return

    seq_names = read_image_sequences(args.train_seqs)
    print(f"[INFO] Load {len(seq_names)} image sequences from {args.train_seqs}.")

    baseline = None

    # records
    best_reward_record, best_iter_record = [], []
    ckp_queue = []
    max_ckp = 3
    best_reward_so_far = -float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Buffer for printing running averages
    from collections import deque
    reward_buf = deque(maxlen=100)
    sparse_buf = deque(maxlen=100)
    kr_buf     = deque(maxlen=100)      # keep ratio buffer

    for epoch in tqdm(range(args.search_epochs)):
        seq_path = np.random.choice(seq_names)
        print(seq_path)

        indices, frames = load_sample_frames(seq_path, frame_interval=args.frame_interval, pil_mode=True)
        # TODO: limit max frames
        if len(frames) > args.max_frame_num:
            # randomly sample a subset of frames
            selected_indices = sorted(np.random.choice(len(frames), args.max_frame_num, replace=False))
            frames = [frames[i] for i in selected_indices]
            indices = [indices[i] for i in selected_indices]


        gt_rgb_map, embedding, pseudo_pcd, pseudo_pcd_conf = infer_sequence(frames, reconstructor, embedding_required=True)
        logits, _ = selector(embedding)

        mask, log_prob, entropy = selector.sample(logits, temp=args.temperature, hard=False)

        # top-k or top-ratio selection, no entropy
        #if args.use_ratio:
        #    mask, log_prob = select_topk(logits, ratio=args.select_ratio, hard=True)
        #else:
        #   mask, log_prob = select_topk(logits, k=args.select_k, hard=True)
        #entropy = torch.zeros_like(log_prob)  # no entropy term if using hard selection

        # Drop the frame which scores less than 0.5
        keep_idx = torch.where(mask.squeeze() > 0.5)[0].cpu().numpy()
        if keep_idx.size == 0:                       # if none selected, select the top-1
            _, top_idx = torch.topk(mask, k=1)
            keep_idx = [top_idx.item()]
        sel_images = [frames[i] for i in keep_idx]

        # Reconstruction and project with selected frames
        dropped_rgb_map, _, pred_pcd, pred_pcd_conf = infer_sequence(sel_images, reconstructor)

        # Reward computation
        sparse = 1.0 - mask.mean()
        reward = compute_reward(dropped_rgb_map, gt_rgb_map[keep_idx]) + args.sparse_coeff * sparse

        baseline, train_info = train_controller(args, selector, optimizer, reward, baseline, log_prob, entropy)

        # Training information
        train_info["sparse"] = sparse.item()
        train_info["keep_ratio"] = (mask > 0.5).float().mean().item()
        train_info["num_select"] = len(keep_idx)
        # monitor buffers
        reward_buf.append(train_info['reward'])
        sparse_buf.append(train_info['sparse'])
        kr_buf.append(train_info['keep_ratio'])

        # train info print
        print(f"[INFO] Epoch {epoch + 1} | "
            f"reward: {train_info['reward']:+.4f} (avg100={sum(reward_buf)/len(reward_buf):+.4f}) | "
            f"keep: {train_info['keep_ratio']:.2%} (avg100={sum(kr_buf)/len(kr_buf):.2%}) | "
            f"sparse: {train_info['sparse']:.4f} (avg100={sum(sparse_buf)/len(sparse_buf):.4f}) | "
            f"entropy: {train_info['entropy']:.3f} | "
            f"loss: {train_info['loss']:+.4f} | "
            f"learning rate: {train_info['lr']:.2e}"
        )

        # save the best checkpoint
        current_r = train_info['reward']
        is_best = current_r > best_reward_so_far
        if is_best:
            best_reward_so_far = current_r

        # save the current checkpoint
        state = {
            'epoch': epoch + 1,
            'controller_state': selector.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'baseline': baseline,
            'reward': current_r,
            'args': args,
        }
        fname = save_checkpoint(state, epoch + 1, current_r, args.save_dir, is_best)

        # maintain ckp queue
        ckp_queue.append((current_r, epoch + 1, fname))
        ckp_queue.sort(key=lambda x: x[0])          # 小顶堆
        cleanup_checkpoints(ckp_queue, max_ckp)

        best_reward_record.append(reward)
        best_iter_record.append(epoch)

        del gt_rgb_map, embedding, logits, mask, log_prob, entropy, dropped_rgb_map, reward

        if epoch % args.val_epoch == 0:
            evaluate(args, selector, reconstructor)


        torch.cuda.empty_cache()
        

        
if __name__ == "__main__":
    args = parse_args()
    main(args)