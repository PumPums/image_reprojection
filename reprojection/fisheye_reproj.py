import math
from typing import Tuple

import numpy as np
import torch
from torch import nn


class FisheyeBackprojDepth(nn.Module):
    """Depth map transformation to a point cloud for Fisheye camera"""

    def __init__(
        self,
        batch_size: int,
        height: int,
        width: int,
        undistort_eps: float = 1e-4,
        undistort_iters: int = 10,
    ):
        super(FisheyeBackprojDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False
        )

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False,
        )

        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0
        )
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False
        )
        self.undistort_eps = undistort_eps
        self.undistort_iters = undistort_iters

    def _unproject_points_batch(
        self,
        radial_params: torch.Tensor,
        xy_origins: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_rays_shape = xy_origins.shape[0]
        # undistort origins
        xy_undistort = radial_and_tangential_undistort(
            xy_origins, radial_params, self.undistort_eps, self.undistort_iters
        )
        # calculate directions
        directions = torch.empty((num_rays_shape,) + (3,), device=device)
        theta = torch.sqrt(torch.sum(xy_undistort**2, dim=-1))
        theta = torch.clip(theta, 0.0, math.pi)
        sin_theta = torch.sin(theta)
        directions[..., 0] = xy_undistort[..., 0] * sin_theta / theta
        directions[..., 1] = xy_undistort[..., 1] * sin_theta / theta
        directions[..., 2] = torch.cos(theta)
        return xy_undistort, directions

    def forward(self, depth, inv_K, radial_distortion=None, RT=None, to_world=False):
        """
        Args:
            depth: depth maps [bs, h, w]
            inv_K: img2cam [bs, 3, 3]
            radial_distortion: radial distortion parameters [bs, 4]
            RT: rotation + translation transformation [Rt] [bs, 3, 4]
            to_world: return unprojected points in world coords
        Return:
            unproj_points: unprojected points from depth map
        """
        device = depth.device

        with torch.no_grad():
            cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords).permute(
                0, 2, 1
            )
            cam_directions = []
            for i in range(self.batch_size):
                _, dirs_batch = self._unproject_points_batch(
                    radial_distortion[i], cam_points[i][..., :2], device
                )
                cam_directions.append(dirs_batch)
            cam_directions = torch.stack(cam_directions, dim=0)

        if to_world:
            R, T = RT[:, :3, :3], RT[:, :3, 3]
            world_directions = cam_directions.bmm(R)
            world_origins = T.expand(world_directions.shape)
            unproj_points = world_origins + world_directions * depth.view(
                self.batch_size, -1, 1
            )
        else:
            unproj_points = depth.view(self.batch_size, -1, 1) * cam_directions
        unproj_points = torch.cat([unproj_points.permute(0, 2, 1), self.ones], 1)
        return unproj_points


class FisheyeProj(nn.Module):
    """3D point cloud projection onto fisheye camera view wrt. intrinsics K and at position [R|T]"""

    def __init__(self, batch_size: int, height: int, width: int, eps: float = 1e-7):
        super(FisheyeProj, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, fo_points, K, distortion=None, RT=None, from_world=False):
        """
        Args:
            fo_points: visible points [bs, n_points, 3]
            K: cam2img [bs, 3, 3]
            radial_distortion: radial distortion parameters [bs, 4]
            RT: rotation + translation transformation [R|T] [bs, 3, 4]
            from_world: return unprojected points in world coords
        Return:
            pix_coords: pix_coords for grid sampling
        """
        device = fo_points.device
        if from_world:
            fo_points = torch.matmul(RT, fo_points)

        # fo_depth = torch.sqrt(fo_points[:, 0, :] ** 2 + fo_points[:, 1, :] ** 2 + fo_points[:, 2, :] ** 2)
        cam_points = []
        for j in range(self.batch_size):
            batch_fo_points = transform_pts_cam_to_fisheye(
                fo_points[j], K[j], distortion[j], device
            )
            cam_points.append(batch_fo_points)

        if self.batch_size == 1:
            cam_points = cam_points[0].unsqueeze(0)
        else:
            cam_points = torch.stack(cam_points, dim=0)

        pix_coords = cam_points[:, :2, :] / (
            cam_points[:, 2, :].unsqueeze(1) + self.eps
        )
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def transform_pts_cam_to_fisheye(pts, intrinsic, distortion, device="cpu"):
    """
    Args:
        pts [torch.tensor]: (3, N) where each column is a 3D point (cam-view)
    Return:
        pts [torch.tensor]: (3, N) , every 3d point location renew for fisheye
    """
    # effect a pitch angle for 3d point cloud to match the fisheye camera
    r = torch.sqrt(torch.pow(pts[0, :], 2) + torch.pow(pts[1, :], 2))
    theta = torch.atan2(r, pts[2, :])
    theta_d = theta * (
        1
        + distortion[0] * torch.pow(theta, 2)
        + distortion[1] * torch.pow(theta, 4)
        + distortion[2] * torch.pow(theta, 6)
        + distortion[3] * torch.pow(theta, 8)
    )
    distort_x = theta_d * pts[0, :] / (r + 1e-8)
    distort_y = theta_d * pts[1, :] / (r + 1e-8)
    nbr_points = pts.shape[1]

    # conversion to pixel coordinates
    distort_points_2d = torch.cat(
        (
            distort_x.reshape(1, -1),
            distort_y.reshape(1, -1),
            torch.ones((1, nbr_points)).to(device),
        )
    )
    points_im = torch.matmul(intrinsic, distort_points_2d)
    return points_im


def radial_and_tangential_undistort(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-3,
    max_iterations: int = 10,
) -> torch.Tensor:
    """Computes undistorted coords given opencv distortion parameters.
    Adapted from MultiNeRF
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509
    https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/camera_utils.py

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].
        eps: The epsilon for the convergence.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The undistorted coordinates.
    """

    # Initialize from the distorted point.
    x = coords[..., 0]
    y = coords[..., 1]

    for _ in range(max_iterations):
        # for radial(4) + tangential(2) distortion use _compute_residual_and_jacobian(...)
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian_simplified(
            x=x,
            y=y,
            xd=coords[..., 0],
            yd=coords[..., 1],
            distortion_params=distortion_params,
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(
            torch.abs(denominator) > eps,
            x_numerator / denominator,
            torch.zeros_like(denominator),
        )
        step_y = torch.where(
            torch.abs(denominator) > eps,
            y_numerator / denominator,
            torch.zeros_like(denominator),
        )

        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)


def _compute_residual_and_jacobian_simplified(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """This is the simplified function only for radial distortion parameters.
    The explanations for this undistortion function are given below.
    """
    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 2]
    k4 = distortion_params[..., 3]
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))
    fx = d * x - xd
    fy = d * y - yd
    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r
    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x
    fx_y = d_y * x
    # Compute derivative of fy over x and y.
    fy_x = d_x * y
    fy_y = d + d_y * y
    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _compute_residual_and_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474
    https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/camera_utils.py

    fisheye624 unsidtortion (radial(6) + tangential(2) + thin(4)):
    https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/camera_utils.py#L632

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].

    distortion_params:
        k1: The first radial distortion parameter.
        k2: The second radial distortion parameter.
        k3: The third radial distortion parameter.
        k4: The fourth radial distortion parameter.
        p1: The first tangential distortion parameter.
        p2: The second tangential distortion parameter.

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
    """

    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 2]
    k4 = distortion_params[..., 3]
    p1 = distortion_params[..., 4]
    p2 = distortion_params[..., 5]

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y
