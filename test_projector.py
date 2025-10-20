# test_projector.py
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# 把你的 reconstructor 文件所在目录加入路径
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
images_list = [Image.open(os.path.join("./images/test_seq", image_name)) \
                for image_name in os.listdir("./images/test_seq")]

recon = SelectedFrameReconstructor(args)

predcitions, _ = recon._vggt_inference(images_list)

images = predcitions["images"]
print(images.device)

# -------------------------------------------------
# 3. 调用并行投影函数
# -------------------------------------------------
with torch.no_grad():
    rgb_r, depth_r, conf_r, mask = recon._project_world_points_to_images(
        images, predcitions["world_points"], predcitions["world_points_conf"], predcitions["extrinsic"], predcitions["intrinsic"])

# -------------------------------------------------
# 4. 快速 sanity check
# -------------------------------------------------
print('-------- output shapes --------')
print('rgb_map  :', rgb_r.shape)      # expect (S,3,H,W)
print('depth_map:', depth_r.shape)    # expect (S,1,H,W)
print('conf_map :', conf_r.shape)     # expect (S,1,H,W)
print('mask_map :', mask.shape)       # expect (S,1,H,W)

print('-------- value range ----------')
print('rgb min/max:', rgb_r.min().item(), '/', rgb_r.max().item())
print('depth min/max:', depth_r.min().item(), '/', depth_r.max().item())
print('conf min/max:', conf_r.min().item(), '/', conf_r.max().item())
print('mask occupancy:', mask.mean().item(), '(>0 表示有有效投影)')

# -------------------------------------------------
# 5. 可视化第一帧（保存到本地）
# -------------------------------------------------
out_dir = Path('_test_render')
out_dir.mkdir(exist_ok=True)

# 原图
for i, image in enumerate(images_list):
    Image.fromarray((images[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))\
        .save(out_dir / f'00{i}_input.png')

    # 重投影
    Image.fromarray((rgb_r[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))\
        .save(out_dir / f'00{i}_reproj.png')

    # 深度伪彩色
    from matplotlib import pyplot as plt
    depth_np = depth_r[0,0].cpu().numpy()
    depth_color = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
    depth_color = (plt.cm.jet(depth_color)[:,:,:3]*255).astype(np.uint8)
    Image.fromarray(depth_color).save(out_dir / f'00{i}_depth.png')

print(f'-------- vis saved to {out_dir.absolute()} --------')

# -------------------------------------------------
# 6. 梯度检查（可选）
# -------------------------------------------------
if True:   # 开关
    images.requires_grad_(True)
    predcitions["world_points"].requires_grad_(True)
    rgb_r, depth_r, conf_r, mask = recon._project_world_points_to_images(
        images, predcitions["world_points"], predcitions["world_points_conf"], predcitions["extrinsic"], predcitions["intrinsic"])

    loss = rgb_r.mean() + depth_r.mean()
    loss.backward()
    assert images.grad is not None and predcitions["world_points"].grad is not None
    print('-------- gradient ok --------')

print('==== All test passed! ====')

