"""Functions for example"""

import json

import numpy as np
import torch

from reprojection import transform_pts_cam_to_fisheye


def calib2np(calib):
    """
    Convert lists to np.array in calib
    """
    new_calib = {}
    for key, val in calib.items():
        if isinstance(val, dict):
            new_calib.update({key: calib2np(val)})
        else:
            new_calib.update({key: np.array(val)})
    return new_calib


def open_calib(path):
    """Read calib file"""
    with open(path, "r", encoding="utf-8") as f:
        calib_ = json.load(f)
    return calib2np(calib_)


def get_4x4tranform(R, T=None):
    """Create R|T tnansform from R[3x3] and T[3x1]"""
    tranform = torch.eye(4, dtype=torch.float64)
    tranform[:3, :3] = torch.from_numpy(R).to(torch.float64)
    if T is not None:
        tranform[:3, 3] = torch.from_numpy(T)[..., 0].to(torch.float64)
    return tranform.unsqueeze(0)


def cloud_projection(points, colors, calib, fisheye=False):
    """Project RGB point cloud onto image view and create depth map and RGB image"""
    R, T, K = calib["R"], calib["T"], calib["K"]
    w_img, h_img = calib["image_size"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    points = (R @ points.T + T).T
    depth = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)

    if fisheye:
        distortion = calib["distortion"]
        points = transform_pts_cam_to_fisheye(
            torch.from_numpy(points.T),
            torch.from_numpy(K),
            torch.from_numpy(distortion),
        ).T.numpy()

    # projection
    proj_img, proj_depth = np.zeros((int(h_img), int(w_img), 3)), np.zeros(
        (int(h_img), int(w_img))
    )

    if fisheye:
        u = points[:, 0].astype(np.int32)
        v = points[:, 1].astype(np.int32)
    else:
        u = (fx * points[:, 0] / points[:, 2] + cx).astype(np.int32)
        v = (fy * points[:, 1] / points[:, 2] + cy).astype(np.int32)

    mask = (u < w_img) & (v < h_img) & (u > 0) & (v > 0)
    u, v = u[mask], v[mask]

    proj_img[v, u, :] = colors[mask]
    proj_depth[v, u] = depth[mask]
    return proj_img[None, ...], proj_depth[None, ...]
