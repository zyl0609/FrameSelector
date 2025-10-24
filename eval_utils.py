import torch
from typing import Dict, List


def preprocess_vggt_predictions(pred_dict: Dict) -> List[torch.Tensor]:
    view1_pose = pred_dict['extrinsic'][0]  #(3, 4) world to camera
    S, H, W, _ = pred_dict['world_points'].shape
    device = pred_dict["world_points"].device
    
    _pred_dict = []
    for i in range(S):
        world_pts = pred_dict['world_points'][i]    # (H, W, 3)
        conf = pred_dict['world_points_conf'][i]    # (H, W)
        R = pred_dict['extrinsic'][i, :3, :3]  # (3, 3)
        t = pred_dict['extrinsic'][i, :3, 3:]  #

        view0_points = torch.matmul(R, world_pts.reshape(-1, 3).T) + t  # (3, H*W)
        view0_points = view0_points.T.reshape(1, H, W, 3)  # add batch dim

        _pred_dict.append(
            {
                "pts3d_in_other_view": view0_points,
                "conf": conf.reshape(1, H, W)
            }
        )

    return _pred_dict