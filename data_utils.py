import torch
from torchvision import transforms as TF
import numpy as np
import cv2
import argparse
import os
import json
from natsort import natsorted
from PIL import Image

from typing import List, Dict, Tuple, Union


def get_video_info(video_path:str)->Dict:
    """
    Reading video info from path.

    :param video_path: the path of video.
    :return: a dictionary of video information: width, height, fps, frame_count, duration.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {}
    info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info['fps'] = cap.get(cv2.CAP_PROP_FPS)
    info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if info['frame_count'] > 0 and info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    else:
        info['duration'] = None
    cap.release()
    return info


def load_sample_frames(
        path:str,
        frame_interval: int = 1,
        pil_mode: bool = False
    )->Tuple[List[int], List[Union[Image.Image, np.ndarray]]]:
    """
    Load frames from image folder or video file.
    
    :param path: The path of image folder or video file.
    :return: A tuple containing a list of frames (as numpy arrays) and a list of corresponding frame indices.
    """

    frames, indices = [], []

    # Process image folder
    if os.path.isdir(path):
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        img_names = natsorted([f for f in os.listdir(str(path))
                            if f.lower().endswith(img_exts)])
        print(f"[INFO] Loading {len(img_names)} images from folder: {os.path.abspath(path)}")
        if not img_names:
            raise ValueError(f'[ERROR] No images found in folder: {path}')
        
        for ind, img_name in enumerate(img_names):
            if ind % frame_interval != 0:
                continue
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('RGB')
            if not pil_mode:
                # to BGR format
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            indices.append(ind)
            frames.append(img)
        return indices, frames
    
    # Process video file
    elif os.path.isfile(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f'[ERROR] Cannot open video: {path}')
        print(f"[INFO] Loading video from file: {os.path.abspath(path)}")
        ind = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if ind % frame_interval == 0:
                if pil_mode:
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                indices.append(ind)
                frames.append(frame)
            ind += 1
        cap.release()
        return indices, frames 
    
    else:
        raise ValueError(f'[ERROR] path must be a video file or image folder: {path}')
    

def load_and_preprocess_sample_frames(sample_frames, *, target_size=518, mode="crop"):
    """
    Slightly modify the code from `vggt.utils.load_fn import load_and_preprocess_images` 
    to directly load images.

        Args:
        sample_frames (list): List of PIL images.
        target_size (int, optional): Target size for both width and height. Defaults to 518.
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(sample_frames) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    #target_size = 518

    # First process all images and collect their shapes
    for img in sample_frames:
        # Open image
        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(sample_frames) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images