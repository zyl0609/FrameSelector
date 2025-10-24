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
    parser.add_argument("--conf_threshold", type=float, default=0.8,
                    help="confidence threshold for data filtering")

    # TODO: add other configurations for data processing here

    
    # frame setting
    #parser.add_argument('--num_slot', type=int, default=50, help='#slots to split the frame sequence')
    parser.add_argument('--infer_seq_size', type=int, default=200, help='max #frames in a single inference')
    parser.add_argument('--max_frame_num', type=int, default=1000, help='max #frames to process in each sequence')
    parser.add_argument('--select_k', type=int, default=500, help='#frames to select')
    parser.add_argument('--select_ratio', type=float, default=0.5, help='ratio of frames to select')
    parser.add_argument('--frame_interval', type=int, default=1, help='take every n-th frame')
    parser.add_argument('--use_ratio', action='store_true', help='use ratio instead of k to select frames')
    # TODO: add other configurations for frame selection here

    
    # training settings
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42, help='global random seed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./ckpt', help='directory to save checkpoints and logs')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='train or eval mode')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint for evaluation or resuming training')
    parser.add_argument('--search_epochs', type=int, default=50)  # 1500
    parser.add_argument('--save_policy_len', type=int, default=10)  # 1500
    parser.add_argument('--val_epoch', type=int, default=10)  # 1500
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/000.pth', help="th pretrain model path.")

    parser.add_argument('--feat_dim', type=int, default=2048)
    parser.add_argument('--controller_hid_size', type=int, default=2048)
    parser.add_argument('--controller_lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sparse_coeff', type=float, default=0.0)
    parser.add_argument('--entropy_coeff', type=float, default=1e-5)
    parser.add_argument('--baseline_decay', type=float, default=0.95)
    parser.add_argument('--controller_grad_clip', type=float, default=10.0)

    # evaluation settings
    parser.add_argument('--eval_interval', type=int, default=1000, help='evaluate every N epochs')
    parser.add_argument('--eval_dataset_path', type=str, default='/opt/ml/code/cvpr2026/data/7scenes', help='Path to the 7Scenes dataset for evaluation')
    parser.add_argument('--eval_output_dir', type=str, default='./mv_recon', help='Path to the 7Scenes dataset for evaluation')

    # teacher model settings
    parser.add_argument('--teacher_name', type=str, default='vggt',
                        choices=['vggt', 'cut3r', 'dust3r'],
                        help='which teacher produces cameras pose & depth map & point-clouds')
    
    # VGGT settings
    parser.add_argument('--vggt_ckpt', type=str, default='./vggt/model.pt')
    parser.add_argument('--vggt_use_point_map', type=bool, default=False, help='use depth map or directly world points')
    parser.add_argument('--vggt_neighbor_size', type=int, default=5, help='size to reconsturct point clouds.')
    parser.add_argument('--vggt_imgsz', type=int, default=518)
    

    

    # TODO: other model settings can be added here

    args = parser.parse_args()
    return args