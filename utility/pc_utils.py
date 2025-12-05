import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def coverage(gt_points, rec_points, distance_threshold=0.01):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)

    cov_cnt = np.count_nonzero(distances <= distance_threshold)

    return cov_cnt / len(gt_points)


def point_cloud_to_volume(points, voxel_size):
    """
    Voxelize a sequential point cloud with average fusion, and return point-to-voxel mapping.

    :param points: torch.Tensor, shape (S, H, W, 3) or (N, 3)
    :param voxel_size: float, voxel size (unit: meters)

    :return fused_points: torch.Tensor, shape (M, 3), voxelized point cloud
    """
    # Input validation
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)

    device, dtype = points.device, points.dtype
    original_shape = points.shape
    is_sequence = points.ndim == 4

    # Flatten to (N, 3)
    if is_sequence:
        S, H, W, _ = original_shape
        points_flat = points.reshape(-1, 3)
        total_points = S * H * W
    else:
        S, H, W = None, None, None
        points_flat = points
        total_points = points.shape[0]

    # Filter invalid points
    valid_mask = torch.isfinite(points_flat).all(dim=1)
    points_valid = points_flat[valid_mask]
    num_valid = points_valid.shape[0]

    # Handle empty point cloud
    if num_valid == 0:
        empty_pts = torch.empty((0, 3), device=device, dtype=dtype)
        empty_counts = torch.empty((0,), device=device, dtype=torch.long)
        map_shape = (S, H, W) if is_sequence else (total_points,)
        empty_map = torch.full(map_shape, -1, device=device, dtype=torch.long)
        return empty_pts, empty_map, empty_counts

    # Voxel indexing
    voxel_indices = torch.floor(points_valid / voxel_size).long()

    # Unique voxels
    unique_voxels, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
    num_voxels = unique_voxels.shape[0]

    # Aggregate coordinates sum and counts
    aggregated_sum = torch.zeros((num_voxels, 3), device=device, dtype=dtype)
    aggregated_count = torch.zeros(num_voxels, device=device, dtype=torch.long)

    inverse_expanded = inverse_indices.unsqueeze(1).expand(-1, 3)
    aggregated_sum.scatter_add_(0, inverse_expanded, points_valid)

    ones = torch.ones_like(inverse_indices, dtype=torch.long)
    aggregated_count.scatter_add_(0, inverse_indices, ones)

    # Average coordinates
    fused_points = aggregated_sum / aggregated_count.float().unsqueeze(1)

    return fused_points


def pcd_to_volume(points, voxel_size, color_required=False):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points = points.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)

    colors = pcd_ds.colors if color_required else None

    return np.asarray(pcd_ds.points), np.asarray(colors)


def icp(
    pred_points,
    gt_points,
    threshold=0.1
):
    
    pr_pts = pred_points.cpu().numpy() \
        if isinstance(pred_points, torch.Tensor) else pred_points
    
    gt_pts = gt_points.cpu().numpy() \
        if isinstance(gt_points, torch.Tensor) else gt_points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pr_pts)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)

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

    return pcd.points, pcd_gt.points