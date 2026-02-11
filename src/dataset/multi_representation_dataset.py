"""Dataset returning multiple representations of each mesh."""

from pathlib import Path
from typing import Callable

import torch

from src.geometry import Mesh3D
from src.dataset.base_modelnet_dataset import BaseModelNetDataset


class MultiRepresentationDataset(BaseModelNetDataset):
    """Dataset returning multiple representations of each mesh.

    Returns a dictionary containing a sampled point cloud alongside
    mesh metadata (vertex and face counts). Useful for models that
    combine geometric features with structural properties.

    Args:
        root_dir: Path to ModelNet directory containing class folders
        split: 'train' or 'test'
        n_points: Number of points to sample from each mesh
        train_ratio: Training fraction when use_existing_split=False
        seed: Random seed for reproducible virtual splits
        transform: Optional callable applied to representation dict
    """

    def __init__(self,
                 root_dir: Path,
                 split: str = 'train',
                 n_points: int = 1024,
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 transform: Callable | None = None):
        self.n_points = n_points
        super().__init__(root_dir, split, use_existing_split=True,
                         train_ratio=train_ratio, seed=seed,
                         transform=transform)

    def _process_mesh(self, mesh: Mesh3D, idx: int) -> dict[str, torch.Tensor]:
        points = mesh.sample_points(n_points=self.n_points)
        return {
            'point_cloud': torch.from_numpy(points).float(),
            'num_vertices': torch.tensor(len(mesh.vertices)),
            'num_faces': torch.tensor(len(mesh.faces)),
        }
