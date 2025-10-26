import torch
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

from PIL import Image
import os


def vis_rgb_maps(rgb_maps, save_dir="./ckpt", indices = [0]):
    os.makedirs(save_dir, exist_ok=True)
    S, C, H, W = rgb_maps.shape
    rgb_maps = rgb_maps.detach()
    for i in indices:
        if i < S:
            rgb_map =  rgb_maps[i].detach().clone()
            rgb_map = rgb_map.permute(1, 2, 0).cpu().numpy()
            rgb_map = (rgb_map * 255).astype(np.uint8)
            img = Image.fromarray(rgb_map)
            img.save(os.path.join(save_dir, f"rgb_map_{i:03d}.png"))



def render_pcd_open3d(
    images: torch.Tensor,            # (K, 3, H, W)  0-1 float32
    world_points: torch.Tensor,      # (K, H, W, 3)
    world_points_conf: torch.Tensor, # (K, H, W)
    extrinsic: torch.Tensor,         # (S, 3, 4)
    intrinsic: torch.Tensor,         # (S, 3, 3)
    conf_threshold: float = 0.5,
):
    """ Visualize point cloud by open3d renderer. """
    K, C, H, W = images.shape
    S = extrinsic.shape[0]
    device = images.device

    # 1. 拼点云（过滤低置信度）
    mask = world_points_conf > conf_threshold
    pts = world_points[mask].cpu().numpy()                  # (N, 3)
    cols = images.permute(0, 2, 3, 1)[mask].cpu().numpy()   # (N, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    # 2. 构造 OffscreenRenderer
    render = o3d.visualization.rendering.OffscreenRenderer(W, H)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    render.scene.add_geometry("pcd", pcd, material)

    rgb_list, depth_list, mask_list = [], [], []
    for s in range(S):
        E = extrinsic[s].cpu().numpy()      # (3,4)
        Kmat = intrinsic[s].cpu().numpy()   # (3,3)
        fx, fy, cx, cy = Kmat[0, 0], Kmat[1, 1], Kmat[0, 2], Kmat[1, 2]

        # PinholeCameraParameters
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        cam.extrinsic = np.vstack([E, [0, 0, 0, 1]])  # 4×4
        render.setup_camera(cam.intrinsic, cam.extrinsic)

        # 渲染
        img  = render.render_to_image()          # o3d.geometry.Image
        depth = render.render_to_depth_image()

        rgb = torch.from_numpy(np.asarray(img)).to(device).float() / 255.0  # (H,W,3)
        dep = torch.from_numpy(np.asarray(depth)).to(device)                # (H,W)
        msk = (dep < 10.0).float()

        rgb_list.append(rgb.permute(2, 0, 1))   # (3,H,W)
        depth_list.append(dep.unsqueeze(0))     # (1,H,W)
        mask_list.append(msk.unsqueeze(0))      # (1,H,W)

    # 3. 堆回 batch
    rgb_map   = torch.stack(rgb_list, 0)
    depth_map = torch.stack(depth_list, 0)
    mask_map  = torch.stack(mask_list, 0)
    return rgb_map, depth_map, mask_map


def render_pcd_open3d_bev(
    images: torch.Tensor,            # (K, 3, H, W)  0-1 float32
    world_points: torch.Tensor,      # (K, H, W, 3)
    world_points_conf: torch.Tensor, # (K, H, W)
    conf_threshold: float = 0.6,
    bev_size: tuple = (1024, 1024),    # 新增：鸟瞰图分辨率 (W, H)
) -> torch.Tensor:                   # 返回 (1, 3, bev_H, bev_W)
    device = images.device
    K, C, H, W = images.shape
    bev_W, bev_H = bev_size           # 解包目标宽高

    # 1. 拼点云
    mask = world_points_conf > conf_threshold
    if mask.sum().item() == 0:
        return torch.ones(1, 3, bev_H, bev_W, device=device) * 0.5

    pts = world_points[mask].cpu().numpy()
    cols = images.permute(0, 2, 3, 1)[mask].cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    # 2. 离线渲染器（目标分辨率）
    render = o3d.visualization.rendering.OffscreenRenderer(bev_W, bev_H)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    render.scene.add_geometry("pcd", pcd, material)

    # 3. 鸟瞰相机（VGGT 坐标系：X右 Y下 Z前）
    scene_min, scene_max = pts.min(0), pts.max(0)
    center = (scene_min + scene_max) / 2
    diag = np.linalg.norm(scene_max - scene_min)

    eye = center + np.array([0, -diag * 1.5, 0])   # -Y 上方
    lookat = center
    up = np.array([0, 0, 1])
    z = (eye - lookat) / np.linalg.norm(eye - lookat)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.stack([x, y, z], axis=1)
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ eye

    # 4. 焦距按分辨率等比例放大（原 512 时 ~650）
    scale = bev_W / 512.0
    fx = fy = 650.0 * scale
    cx, cy = bev_W / 2, bev_H / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(bev_W, bev_H, fx, fy, cx, cy)
    render.setup_camera(intrinsic, extrinsic)

    # 5. 渲染
    img_o3d = render.render_to_image()
    rgb = torch.from_numpy(np.asarray(img_o3d)).to(device).float() / 255.0
    return rgb.permute(2, 0, 1).unsqueeze(0)   # (1, 3, bev_H, bev_W)