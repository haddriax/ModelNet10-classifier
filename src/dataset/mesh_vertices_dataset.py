"""Dataset returning raw mesh vertices (padded/truncated to fixed size)."""

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.geometry import Mesh3D
from src.dataset.base_modelnet_dataset import BaseModelNetDataset


class MeshVerticesDataset(BaseModelNetDataset):
    """Dataset returning raw mesh vertices (padded/truncated).

    Returns vertex coordinates directly from the mesh, padded with zeros
    or truncated to a fixed size. Useful for models that operate on
    raw vertex data rather than sampled point clouds.

    Args:
        root_dir: Path to ModelNet directory containing class folders
        split: 'train' or 'test'
        max_vertices: Maximum number of vertices to return
        train_ratio: Training fraction when use_existing_split=False
        seed: Random seed for reproducible virtual splits
        transform: Optional callable applied to vertex tensors
    """

    def __init__(self,
                 root_dir: Path,
                 split: str = 'train',
                 max_vertices: int = 2048,
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 transform: Callable | None = None):
        self.max_vertices = max_vertices
        super().__init__(root_dir, split, use_existing_split=True,
                         train_ratio=train_ratio, seed=seed,
                         transform=transform)

    def _process_mesh(self, mesh: Mesh3D, idx: int) -> torch.Tensor:
        vertices = mesh.vertices
        n_verts = len(vertices)

        if n_verts >= self.max_vertices:
            vertices = vertices[:self.max_vertices]
        else:
            padding = np.zeros((self.max_vertices - n_verts, 3))
            vertices = np.vstack([vertices, padding])

        return torch.from_numpy(vertices).float()
