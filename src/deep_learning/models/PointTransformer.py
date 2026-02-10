"""Point Transformer v1 for point cloud classification.

Paper: "Point Transformer"
       Zhao et al. 2021, ICCV (arXiv:2012.09164)

Architecture:
    5-stage encoder with vector self-attention and transition down layers.
    Relative position encoding via MLP on coordinate differences.
    Pure PyTorch implementation — no custom CUDA kernels.
"""

import torch
from torch import nn
import torch.nn.functional as F

from src.deep_learning.models.PointNetPP import farthest_point_sample, index_points


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def knn_points(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Find k nearest neighbors.

    Args:
        xyz: Source points [B, N, 3]
        new_xyz: Query points [B, M, 3]
        k: Number of neighbors

    Returns:
        Neighbor indices [B, M, k]
    """
    dist = -2 * torch.matmul(new_xyz, xyz.permute(0, 2, 1))  # [B, M, N]
    dist += torch.sum(new_xyz ** 2, -1, keepdim=True)  # [B, M, 1]
    dist += torch.sum(xyz ** 2, -1).unsqueeze(1)  # [B, 1, N]
    _, idx = dist.topk(k, dim=-1, largest=False)  # [B, M, k]
    return idx


# ---------------------------------------------------------------------------
# Point Transformer Layer
# ---------------------------------------------------------------------------

class PointTransformerLayer(nn.Module):
    """Vector self-attention on point clouds.

    Computes attention weights as vectors (not scalars) that modulate
    individual feature channels. Uses subtraction relation and relative
    position encoding.

    Args:
        dim: Feature dimension
        k: Number of nearest neighbors
    """

    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.k = k

        # Q, K, V projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Relative position encoding: 3D coords → feature dim
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # Attention weight encoding
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            xyz: Point coordinates [B, N, 3]
            features: Point features [B, N, C]

        Returns:
            Updated features [B, N, C]
        """
        B, N, C = features.shape

        # kNN
        idx = knn_points(xyz, xyz, self.k)  # [B, N, k]

        # Q, K, V
        q = self.to_q(features)  # [B, N, C]
        k = self.to_k(features)  # [B, N, C]
        v = self.to_v(features)  # [B, N, C]

        # Gather neighbor K, V
        k_grouped = index_points(k, idx)  # [B, N, k, C]
        v_grouped = index_points(v, idx)  # [B, N, k, C]

        # Relative position encoding
        xyz_grouped = index_points(xyz, idx)  # [B, N, k, 3]
        pos_diff = xyz.unsqueeze(2) - xyz_grouped  # [B, N, k, 3]
        delta = self.pos_mlp(pos_diff)  # [B, N, k, C]

        # Vector attention: subtraction relation
        q_expanded = q.unsqueeze(2).expand_as(k_grouped)  # [B, N, k, C]
        attn = self.attn_mlp(q_expanded - k_grouped + delta)  # [B, N, k, C]
        attn = F.softmax(attn, dim=2)  # softmax over neighbors

        # Weighted aggregation
        out = torch.sum(attn * (v_grouped + delta), dim=2)  # [B, N, C]

        return out


# ---------------------------------------------------------------------------
# Point Transformer Block (with residual connection)
# ---------------------------------------------------------------------------

class PointTransformerBlock(nn.Module):
    """Transformer block: pre-MLP → attention → post-MLP + residual.

    Args:
        dim: Feature dimension
        k: Number of nearest neighbors
    """

    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )
        self.attn = PointTransformerLayer(dim, k)
        self.post = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            xyz: Point coordinates [B, N, 3]
            features: Point features [B, N, C]

        Returns:
            Updated features [B, N, C]
        """
        B, N, C = features.shape
        residual = features

        # Pre-processing (BatchNorm expects [B, C] or [B, C, ...])
        x = self.pre[0](features)  # Linear
        x = self.pre[1](x.transpose(1, 2)).transpose(1, 2)  # BN on [B, C, N]
        x = self.pre[2](x)  # ReLU

        # Attention
        x = self.attn(xyz, x)

        # Post-processing
        x = self.post[0](x)  # Linear
        x = self.post[1](x.transpose(1, 2)).transpose(1, 2)  # BN
        x = self.post[2](x)  # ReLU

        return x + residual


# ---------------------------------------------------------------------------
# Transition Down (downsampling)
# ---------------------------------------------------------------------------

class TransitionDown(nn.Module):
    """Downsample points via FPS + kNN local pooling.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        npoint: Number of output points
        k: Number of neighbors for local grouping
    """

    def __init__(self, in_dim: int, out_dim: int, npoint: int, k: int = 16):
        super().__init__()
        self.npoint = npoint
        self.k = k

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            xyz: Point coordinates [B, N, 3]
            features: Point features [B, N, C_in]

        Returns:
            new_xyz: Downsampled coordinates [B, npoint, 3]
            new_features: Downsampled features [B, npoint, C_out]
        """
        B, N, C = features.shape

        # FPS to select centroids
        fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]

        # kNN for each centroid
        idx = knn_points(xyz, new_xyz, self.k)  # [B, npoint, k]
        grouped_features = index_points(features, idx)  # [B, npoint, k, C_in]

        # MLP on each neighbor
        # Apply linear
        grouped_features = self.mlp[0](grouped_features)  # [B, npoint, k, C_out]
        # Apply BN: reshape to [B*npoint*k, C_out] for BN1d
        shape = grouped_features.shape
        grouped_features = self.mlp[1](grouped_features.reshape(-1, shape[-1])).reshape(shape)
        grouped_features = self.mlp[2](grouped_features)  # ReLU

        # Max pool over neighbors
        new_features = grouped_features.max(dim=2)[0]  # [B, npoint, C_out]

        return new_xyz, new_features


# ---------------------------------------------------------------------------
# Full classification network
# ---------------------------------------------------------------------------

class PointTransformer(nn.Module):
    """Point Transformer v1 classification network.

    5-stage encoder with vector self-attention at each stage,
    transition-down layers for hierarchical feature learning.

    Args:
        num_classes: Number of output classes
        k: Number of nearest neighbors for attention and grouping
        dropout: Dropout probability in classifier
    """

    def __init__(self, num_classes: int = 10, k: int = 16, dropout: float = 0.5):
        super().__init__()

        # Input embedding
        self.input_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # Stage 1: 1024 pts, 32 dims
        self.block1 = PointTransformerBlock(32, k)
        self.td1 = TransitionDown(32, 64, npoint=256, k=k)

        # Stage 2: 256 pts, 64 dims
        self.block2 = PointTransformerBlock(64, k)
        self.td2 = TransitionDown(64, 128, npoint=64, k=k)

        # Stage 3: 64 pts, 128 dims
        self.block3 = PointTransformerBlock(128, k)
        self.td3 = TransitionDown(128, 256, npoint=16, k=k)

        # Stage 4: 16 pts, 256 dims
        self.block4 = PointTransformerBlock(256, k)
        self.td4 = TransitionDown(256, 512, npoint=4, k=min(k, 4))

        # Stage 5: 4 pts, 512 dims
        self.block5 = PointTransformerBlock(512, k=min(k, 4))

        # Classifier head
        self.classifier = nn.Sequential(
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

        # Input embedding
        B, N, _ = x.shape
        features = self.input_mlp[0](x)  # Linear: [B, N, 32]
        features = self.input_mlp[1](features.transpose(1, 2)).transpose(1, 2)  # BN
        features = self.input_mlp[2](features)  # ReLU

        # Stage 1
        features = self.block1(xyz, features)  # [B, 1024, 32]
        xyz, features = self.td1(xyz, features)  # [B, 256, 3], [B, 256, 64]

        # Stage 2
        features = self.block2(xyz, features)  # [B, 256, 64]
        xyz, features = self.td2(xyz, features)  # [B, 64, 3], [B, 64, 128]

        # Stage 3
        features = self.block3(xyz, features)  # [B, 64, 128]
        xyz, features = self.td3(xyz, features)  # [B, 16, 3], [B, 16, 256]

        # Stage 4
        features = self.block4(xyz, features)  # [B, 16, 256]
        xyz, features = self.td4(xyz, features)  # [B, 4, 3], [B, 4, 512]

        # Stage 5
        features = self.block5(xyz, features)  # [B, 4, 512]

        # Global average pooling
        x = features.mean(dim=1)  # [B, 512]

        # Classification
        x = self.classifier(x)  # [B, num_classes]
        return x
