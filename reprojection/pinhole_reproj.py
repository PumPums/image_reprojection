import numpy as np
import torch
from torch import nn


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    Monodepth2 emplementation
    https://github.com/nianticlabs/monodepth2/blob/master/layers.py
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

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

    def forward(self, depth, inv_K):
        """
        Args:
            depth: depth maps [bs, h, w]
            inv_K: img2cam [bs, 3, 3]
        Return:
            cam_points: unprojected points from depth map
        """
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T."""

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, RT):
        """
        Args:
            points: visible points [bs, n_points, 3]
            K: cam2img [bs, 3, 3]
            RT: rotation + translation transformation [R|T] [bs, 3, 4]
        Return:
            ix_coords [torch.tensor]: pix_coords for grid sampling
        """
        P = torch.matmul(K, RT)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (
            cam_points[:, 2, :].unsqueeze(1) + self.eps
        )
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords
