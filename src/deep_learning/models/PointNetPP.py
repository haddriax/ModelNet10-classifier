"""PointNet++ (SSG) for point cloud classification.

Paper: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
       Qi et al. 2017, arXiv:1706.02413

Architecture:
    3 Set Abstraction layers with Single Scale Grouping (SSG).
    Farthest Point Sampling + Ball Query + local PointNet at each level.
    Pure PyTorch implementation — no custom CUDA kernels.
"""

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility functions (pure PyTorch)
# ---------------------------------------------------------------------------

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
        src: Source points [B, N, 3]
        dst: Target points [B, M, 3]

    Returns:
        Squared distances [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative farthest point sampling.

    Args:
        xyz: Point cloud [B, N, 3]
        npoint: Number of points to sample

    Returns:
        Sampled point indices [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by indices.

    Args:
        points: Input points [B, N, C]
        idx: Indices [B, S] or [B, S, K]

    Returns:
        Indexed points [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Ball query: find all points within radius of each centroid.

    Args:
        radius: Ball query radius
        nsample: Maximum number of samples per group
        xyz: All points [B, N, 3]
        new_xyz: Query centroids [B, S, 3]

    Returns:
        Group indices [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # Pad incomplete groups by repeating first index
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


# ---------------------------------------------------------------------------
# Set Abstraction module
# ---------------------------------------------------------------------------

class PointNetSetAbstraction(nn.Module):
    """Set Abstraction layer: FPS + Ball Query + local PointNet.

    Args:
        npoint: Number of centroids to sample (None for global)
        radius: Ball query radius
        nsample: Max samples per group
        in_channel: Input feature channels (includes xyz=3)
        mlp: List of output channels for shared MLP
        group_all: If True, group all points (global feature)
    """

    def __init__(self, npoint: int | None, radius: float | None, nsample: int | None,
                 in_channel: int, mlp: list[int], group_all: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            xyz: Point coordinates [B, N, 3]
            points: Point features [B, N, D] or None

        Returns:
            new_xyz: Sampled centroids [B, npoint, 3]
            new_points: Aggregated features [B, npoint, C_out]
        """
        if self.group_all:
            new_xyz, new_points = self._sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self._sample_and_group(xyz, points)

        # new_points: [B, npoint, nsample, C]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C, nsample, npoint]

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # [B, C_out, npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, C_out]

        return new_xyz, new_points

    def _sample_and_group(self, xyz: torch.Tensor, points: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """FPS + ball query grouping."""
        fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # [B, npoint, nsample]
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # Normalize to centroid

        if points is not None:
            grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
        else:
            new_points = torch.cat([grouped_xyz_norm, grouped_xyz], dim=-1)  # [B, npoint, nsample, 3+3]

        return new_xyz, new_points

    def _sample_and_group_all(self, xyz: torch.Tensor, points: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Group all points into a single set (global feature)."""
        device = xyz.device
        B, N, C = xyz.shape

        new_xyz = torch.zeros(B, 1, C, device=device)
        grouped_xyz = xyz.view(B, 1, N, C)

        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz

        return new_xyz, new_points


# ---------------------------------------------------------------------------
# Full classification network
# ---------------------------------------------------------------------------

class PointNetPP(nn.Module):
    """PointNet++ SSG classification network.

    Args:
        num_classes: Number of output classes
        dropout: Dropout probability in classifier
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        # Set Abstraction layers (SSG)
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3 + 3, mlp=[64, 64, 128],
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024],
            group_all=True,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Point cloud [B, N, 3]

        Returns:
            Class logits [B, num_classes]
        """
        xyz = x  # [B, N, 3]
        points = None

        # Hierarchical feature learning
        xyz, points = self.sa1(xyz, points)   # [B, 512, 3], [B, 512, 128]
        xyz, points = self.sa2(xyz, points)   # [B, 128, 3], [B, 128, 256]
        xyz, points = self.sa3(xyz, points)   # [B, 1, 3],   [B, 1, 1024]

        x = points.view(points.size(0), -1)   # [B, 1024]
        x = self.classifier(x)                # [B, num_classes]
        return x
