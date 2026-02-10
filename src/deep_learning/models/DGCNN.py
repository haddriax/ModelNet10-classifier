"""Dynamic Graph CNN for point cloud classification.

Paper: "Dynamic Graph CNN for Learning on Point Clouds"
       Wang et al. 2019, arXiv:1801.07829

Architecture:
    4 EdgeConv layers with dynamic k-NN graph recomputation in feature space.
    Multi-scale feature aggregation via concatenation.
    Dual global pooling (max + avg).
"""

import torch
from torch import nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k nearest neighbors in feature space.

    Args:
        x: Point features [B, C, N]
        k: Number of nearest neighbors

    Returns:
        idx: Neighbor indices [B, N, k]
    """
    # Pairwise squared distances: ||x_i - x_j||^2
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # [B, N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [B, 1, N]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [B, N, N] (negated)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [B, N, k]
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: torch.Tensor | None = None) -> torch.Tensor:
    """Build edge features from k-NN graph.

    For each point x_i and its neighbor x_j, constructs edge features
    by concatenating [x_i, x_j - x_i].

    Args:
        x: Point features [B, C, N]
        k: Number of nearest neighbors
        idx: Pre-computed neighbor indices [B, N, k], or None to compute

    Returns:
        Edge features [B, 2C, N, k]
    """
    batch_size, num_dims, num_points = x.size()
    device = x.device

    if idx is None:
        idx = knn(x, k=k)  # [B, N, k]

    # Flatten batch indices for gathering
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base  # [B, N, k] with global indices
    idx = idx.view(-1)  # [B*N*k]

    x = x.transpose(2, 1).contiguous()  # [B, N, C]
    feature = x.view(batch_size * num_points, -1)[idx, :]  # [B*N*k, C]
    feature = feature.view(batch_size, num_points, k, num_dims)  # [B, N, k, C]

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # [B, N, k, C]

    # Edge features: concat(x_i, x_j - x_i)
    feature = torch.cat((feature - x, x), dim=3)  # [B, N, k, 2C]
    feature = feature.permute(0, 3, 1, 2).contiguous()  # [B, 2C, N, k]

    return feature


class DGCNN(nn.Module):
    """Dynamic Graph CNN for point cloud classification.

    Args:
        num_classes: Number of output classes
        k: Number of nearest neighbors for EdgeConv
        emb_dims: Embedding dimension after feature aggregation
        dropout: Dropout probability in classifier
    """

    def __init__(self, num_classes: int = 10, k: int = 20, emb_dims: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Feature aggregation
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(emb_dims * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
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
        batch_size = x.size(0)
        x = x.transpose(2, 1)  # [B, 3, N]

        # EdgeConv block 1
        x = get_graph_feature(x, k=self.k)  # [B, 6, N, k]
        x = self.conv1(x)  # [B, 64, N, k]
        x1 = x.max(dim=-1)[0]  # [B, 64, N]

        # EdgeConv block 2 (dynamic graph in feature space)
        x = get_graph_feature(x1, k=self.k)  # [B, 128, N, k]
        x = self.conv2(x)  # [B, 64, N, k]
        x2 = x.max(dim=-1)[0]  # [B, 64, N]

        # EdgeConv block 3
        x = get_graph_feature(x2, k=self.k)  # [B, 128, N, k]
        x = self.conv3(x)  # [B, 128, N, k]
        x3 = x.max(dim=-1)[0]  # [B, 128, N]

        # EdgeConv block 4
        x = get_graph_feature(x3, k=self.k)  # [B, 256, N, k]
        x = self.conv4(x)  # [B, 256, N, k]
        x4 = x.max(dim=-1)[0]  # [B, 256, N]

        # Multi-scale feature aggregation
        x = torch.cat((x1, x2, x3, x4), dim=1)  # [B, 512, N]
        x = self.conv5(x)  # [B, emb_dims, N]

        # Dual global pooling
        x_max = x.max(dim=-1)[0]  # [B, emb_dims]
        x_avg = x.mean(dim=-1)  # [B, emb_dims]
        x = torch.cat((x_max, x_avg), dim=1)  # [B, emb_dims*2]

        # Classification
        x = self.classifier(x)  # [B, num_classes]
        return x
