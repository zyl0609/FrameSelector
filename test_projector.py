# test_projector.py
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# 把你的 reconstructor 文件所在目录加入路径
from data_utils import *
from vis_utils import *
sys.path.append(str(Path(__file__).parent))
from frame_recon import SelectedFrameReconstructor   # ← 改成实际文件名
from config import *

# -------------------------------------------------
# 2. 实例化 reconstructor（只借 projector，不加载 VGGT）
# -------------------------------------------------
args = parse_args()
args.teacher_name = 'vggt'   # 设置教师模型名称
args.vggt_ckpt = './vggt/model.pt'  # 设置 VGGT 权重路径
args.device = 'cuda'
args.vggt_imgsz = 518

sys.path.append("./vggt")


from PIL import Image
import os
images_names = sorted([os.path.join("./images/test_seq", image_name) \
                for image_name in os.listdir("./images/test_seq")])
images_list = [Image.open(img_path).convert('RGB') for img_path in images_names]





recon = SelectedFrameReconstructor(args)
predictions, _ = recon._vggt_inference(images_list)
recon.free_image_cache()
#predictions_, _ = recon._vggt_inference(images_list[0:3])


for i in range(len(images_list)):
    if i == 0:
        predictions_, _ = recon._vggt_inference(images_list[i:i+1])
    else:
        pred, _ = recon._vggt_inference(images_list[i:i+1])

        for key in pred.keys():
            if isinstance(pred[key], torch.Tensor):
                predictions_[key] = torch.cat([predictions[key], pred[key]], dim=0)

recon.free_image_cache()
rgb_r, depth_r, conf_map, *_ = recon.project_world_points_to_images(
    predictions_["images"],
    predictions_["world_points"],
    predictions_["world_points_conf"],
    predictions["extrinsic"],
    predictions["intrinsic"],
    0.5
)
print(predictions_["images"].shape)

pseudo_gt_rgb, pseudo_gt_depth, conf_gt_map, *_ = recon.project_world_points_to_images(
    predictions["images"],
    predictions["world_points"],
    predictions["world_points_conf"],
    predictions["extrinsic"],
    predictions["intrinsic"],
    0.5
)

# test
#rgb_out, *_ = render_pcd_open3d(
#    predictions["images"],
#    predictions["world_points"],
#    predictions["world_points_conf"],
#    predictions["extrinsic"],
#    predictions["intrinsic"],
#)
rgb_out = render_pcd_open3d_bev(
    predictions_["images"],
    predictions_["world_points"],
    predictions_["world_points_conf"]
)
print(rgb_out.shape)

from reward_utils import ssim_loss
ssim = ssim_loss(rgb_r, pseudo_gt_rgb, window_size=11)
l1   = torch.nn.functional.l1_loss(rgb_r, pseudo_gt_rgb)
print(f"L1        = {l1.item():.4f}")
print(f"SSIM-loss = {ssim.item():.4f}")

#from frame_recon import infer_sequence
#rgb_r, _, emb = infer_sequence(images_list, recon, seq_size=1, embedding_required=True)



S, H, W = predictions["world_points_conf"].shape
print(torch.count_nonzero(predictions["world_points_conf"] > 0.9).item() / (S*H*W))


# -------------------------------------------------
# 4. 快速 sanity check
# -------------------------------------------------
print('-------- output shapes --------')
print("len(images_list):", len(images_list))
print('rgb_map  :', rgb_r.shape)      # expect (S,3,H,W)


print('-------- value range ----------')
print('rgb min/max:', rgb_r.min().item(), '/', rgb_r.max().item())
print('conf min/max:', predictions["world_points_conf"].min().item(), '/', conf_map.max().item())


def save_rgb_maps(rgb_maps, save_dir="./output_rgb_maps", name=""):
    os.makedirs(save_dir, exist_ok=True)
    S, C, H, W = rgb_maps.shape
    for i in range(S):
        rgb_map = rgb_maps[i].permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, 3)
        rgb_map = (rgb_map * 255).astype(np.uint8)  # 转换为 uint8
        img = Image.fromarray(rgb_map)
        img.save(os.path.join(save_dir, f"rgb_map_{name}_{i:03d}.png"))
    print(f"Saved {S} RGB maps to {save_dir}")

save_rgb_maps(pseudo_gt_rgb, name='gt')  
save_rgb_maps(rgb_r, name='sparse')
save_rgb_maps(rgb_out, name='batch')

