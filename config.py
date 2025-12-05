import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Data Processing Configurations")

    # dataset settings
    parser.add_argument('--dataset', type=str, default='7scenes',
                    choices=['7scenes', 'neural-rgbd', 'tum'],
                    help='which 3D reconstruction benchmark to use')
    parser.add_argument('--train_seqs', type=str, default="./7scenes_seq.txt",
                    help='absolute path to train sequences file')
    parser.add_argument('--val_seqs', type=str, default="./val_seqs.txt",
                    help='absolute path to validation sequences file')
    parser.add_argument("--pcd_conf_thresh", type=float, default=0.0,
                    help="confidence threshold for data filtering")

    # TODO: add other configurations for data processing here

    
    # frame setting
    parser.add_argument('--infer_seq_size', type=int, default=1000, help='max #frames in a single inference')
    parser.add_argument('--max_frame_num', type=int, default=1000, help='max #frames to process in each sequence')
    parser.add_argument('--select_k', type=int, default=500, help='#frames to select')
    parser.add_argument('--select_ratio', type=float, default=0.1, help='ratio of frames to select')
    parser.add_argument('--frame_interval', type=int, default=1, help='take every n-th frame')
    #parser.add_argument('--use_ratio', action='store_true', help='use ratio instead of k to select frames')
    parser.add_argument('--use_dropped', action='store_true', help='use dropped frames as pseudo-ground truth.')
    # TODO: add other configurations for frame selection here


    # frame selector/controller settings
    parser.add_argument('--feat_size', type=int, default=512)
    parser.add_argument('--slot_sz', type=int, default=10)
    parser.add_argument('--controller_hid_size', type=int, default=256)
    
    
    # training settings
    parser.add_argument('--warmup_lr', type=float, default=1e-3)
    parser.add_argument('--controller_lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--sparse_coeff', type=float, default=1e-3)
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--baseline_decay', type=float, default=0.9)
    parser.add_argument('--controller_grad_clip', type=float, default=10.0)

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42, help='global random seed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./ckpt', help='directory to save checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint for evaluation or resuming training')
    parser.add_argument('--search_epochs', type=int, default=1000)
    parser.add_argument('--val_epoch', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/best.pth', help="th pretrain model path.")

    # evaluation settings
    parser.add_argument('--eval_dataset_path', type=str, default='/opt/ml/code/cvpr2026/data/7scenes', help='Path to the 7Scenes dataset for evaluation')
    parser.add_argument('--eval_output_dir', type=str, default='./results/eval_res', help='Path to the 7Scenes dataset for evaluation')

    # teacher model settings
    parser.add_argument('--teacher_name', type=str, default='fast_vggt',
                        choices=['vggt', 'm_vggt', 'fast_vggt', 'cut3r', 'dust3r'],
                        help='which teacher produces cameras pose & depth map & point-clouds')
    
    # VGGT settings
    parser.add_argument('--vggt_ckpt', type=str, default='./pretrained/model.pt')
    parser.add_argument('--vggt_use_point_map', type=bool, default=True, help='use depth map or directly world points')
    parser.add_argument('--vggt_neighbor_size', type=int, default=2, help='size to reconsturct point clouds.')
    parser.add_argument('--vggt_imgsz', type=int, default=518)

    parser.add_argument('--m_vggt_ckpt', type=str, default='./pretrained/model_tracker_fixed_e20.pt')

    # Viser setting
    parser.add_argument('--port', type=int, default=8080)

    

    # TODO: other model settings can be added here

    args = parser.parse_args()
    return args