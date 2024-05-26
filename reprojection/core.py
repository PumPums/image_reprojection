from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def reproject_images(
    depth_maps_curr: Tensor,
    rgb_imgs_prev: Tensor,
    batch_K: Tensor,
    batch_frame_transform: Tensor,
    batch_ego2cam: Tensor,
    batch_cam2ego: Tensor,
    proj: Callable,
    back_proj: Callable,
    batch_distortion: Tensor = torch.zeros(4),
    is_fisheye: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Function to reproject previous frame RGB image to current frame"""
    N = depth_maps_curr.shape[0]
    reproj_rgb, reproj_masks = [], []

    for i in range(N):
        # get transforms
        K = batch_K[i].unsqueeze(0)
        K_inv = K.inverse()
        ego2cam = batch_ego2cam[i].unsqueeze(0)
        cam2ego = batch_cam2ego[i].unsqueeze(0)

        # update RT transform wrt ego2cam and cam2ego
        frame_RT = (cam2ego @ batch_frame_transform @ ego2cam).to(torch.float32)

        # get maps and imgs
        depth_map_curr = depth_maps_curr[i]
        rgb_img_prev = rgb_imgs_prev[i]
        mask = depth_map_curr != 0

        # apply projs
        if is_fisheye:
            distortion = batch_distortion[i].unsqueeze(0)
            cam_points = back_proj(depth_map_curr, K_inv, distortion)
            pix_coords = proj(cam_points, K, distortion, frame_RT[:, :3, :], True)
        else:
            cam_points = back_proj(depth_map_curr, K_inv)
            pix_coords = proj(cam_points, K, frame_RT[:, :3, :])

        # find projected colours
        input_color_after_mask = (rgb_img_prev * mask).unsqueeze(0)
        reproject_out_color = F.grid_sample(
            input_color_after_mask,
            pix_coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        reproject_mask = (reproject_out_color != 0).min(1)[0]
        reproj_rgb.append(reproject_out_color)
        reproj_masks.append(reproject_mask)

    reproj_rgb = torch.cat(reproj_rgb, dim=0)
    reproj_masks = torch.cat(reproj_masks, dim=0)
    return reproj_rgb, reproj_masks
