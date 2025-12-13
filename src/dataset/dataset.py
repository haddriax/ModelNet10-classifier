from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Optional
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.builders.mesh_3D_builder import Mesh3DBuilder
from src.geometry.Mesh_3D import Mesh3D, Sampling


class BaseModelNetDataset(Dataset, ABC):
    """Abstract base dataset for ModelNet10/40 with virtual train/test split"""

    def __init__(self,
                 root_dir: Path,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 transform: Optional[callable] = None):
        """
        Args:
            root_dir: Path to ModelNet directory (contains class folders)
            split: 'train' or 'test'
            train_ratio: Fraction of data for training (0.0 to 1.0)
            seed: Random seed for reproducible splits
            transform: Optional transformation to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.transform = transform

        self.files = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._build_index()

    def _build_index(self):
        """Load files from existing train/test folders and optionally resplit"""
        all_files = []
        all_labels = []

        classes: list[str] = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        print(f"[{self.__class__.__name__}] {self.split} set: {len(classes)} classes")

        for idx, class_name in enumerate(classes):
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
                self.idx_to_class[self.class_to_idx[class_name]] = class_name

            class_idx = self.class_to_idx[class_name]

            # class_dir = root dir + class
            class_dir: Path = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found at'{class_dir.resolve()}'")
            if not class_dir.is_dir():
                raise NotADirectoryError(f"Class path is not a directory at '{class_dir.resolve()}'")

            # class_dire now = root dir + class + split
            class_dir = class_dir / self.split

            if not class_dir.exists():
                raise FileNotFoundError(f"No split '{self.split}' found for {class_name} at '{class_dir.resolve()}'")



            class_files = list(class_dir.glob('*.off'))
            all_files.extend(class_files)
            all_labels.extend([class_idx] * len(class_files))

            print(f"[{self.__class__.__name__}] [{class_name}{self.split}] with {len(class_files)} samples")

        # Create stratified split
        random.seed(self.seed)

        for class_idx in range(len(self.class_to_idx)):
            class_indices = [i for i, label in enumerate(all_labels) if label == class_idx]
            random.shuffle(class_indices)

            split_point = int(len(class_indices) * self.train_ratio)
            indices = class_indices[:split_point] if self.split == 'train' else class_indices[split_point:]

            for idx in indices:
                self.files.append(all_files[idx])
                self.labels.append(all_labels[idx])

        print(f"[{self.__class__.__name__}] {self.split}: {len(self.files)} samples "
              f"from {len(self.class_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.files)

    @abstractmethod
    def _process_mesh(self, mesh: Mesh3D) -> torch.Tensor:
        """Process mesh into desired representation"""
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        mesh = Mesh3DBuilder.from_off_file(self.files[idx])
        data = self._process_mesh(mesh)

        if self.transform:
            data = self.transform(data)

        return data, self.labels[idx]

    def get_class_name(self, idx: int) -> str:
        return self.idx_to_class[idx]


class PointCloudDataset(BaseModelNetDataset):
    """Dataset returning sampled point clouds"""

    def __init__(self,
                 root_dir: Path,
                 split: str = 'train',
                 n_points: int = 1024,
                 sampling_method: Sampling = Sampling.UNIFORM,
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 transform: Optional[callable] = None):
        self.n_points = n_points
        self.sampling_method = sampling_method
        super().__init__(root_dir, split, train_ratio, seed, transform)

    def _process_mesh(self, mesh: Mesh3D) -> torch.Tensor:
        pts = mesh.sample_points(n_points=self.n_points, method=self.sampling_method)
        return torch.from_numpy(pts).float()


class MeshVerticesDataset(BaseModelNetDataset):
    """Dataset returning raw mesh vertices (padded/truncated)"""

    def __init__(self,
                 root_dir: Path,
                 split: str = 'train',
                 max_vertices: int = 2048,
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 transform: Optional[callable] = None):
        self.max_vertices = max_vertices
        super().__init__(root_dir, split, train_ratio, seed, transform)

    def _process_mesh(self, mesh: Mesh3D) -> torch.Tensor:
        vertices = mesh.vertices
        n_verts = len(vertices)

        if n_verts >= self.max_vertices:
            vertices = vertices[:self.max_vertices]
        else:
            padding = np.zeros((self.max_vertices - n_verts, 3))
            vertices = np.vstack([vertices, padding])

        return torch.from_numpy(vertices).float()


class MultiRepresentationDataset(BaseModelNetDataset):
    """Dataset returning multiple representations"""

    def __init__(self,
                 root_dir: Path,
                 split: str = 'train',
                 n_points: int = 1024,
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 transform: Optional[callable] = None):
        self.n_points = n_points
        super().__init__(root_dir, split, train_ratio, seed, transform)

    def _process_mesh(self, mesh: Mesh3D) -> Dict[str, torch.Tensor]:
        points = mesh.sample_points(n_points=self.n_points)
        return {
            'point_cloud': torch.from_numpy(points).float(),
            'num_vertices': torch.tensor(len(mesh.vertices)),
            'num_faces': torch.tensor(len(mesh.faces)),
        }

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        mesh = Mesh3DBuilder.from_off_file(self.files[idx])
        data = self._process_mesh(mesh)

        if self.transform:
            data = self.transform(data)

        return data, self.labels[idx]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_dir = Path(r"../../data/ModelNet10/models")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir.resolve()} does not exist.")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory {data_dir.resolve()} is not a directory.")
    if not any(data_dir.iterdir()):
        raise ValueError(f"Data directory {data_dir.resolve()} is empty.")

    train_dataset = PointCloudDataset(root_dir=data_dir, split='train', n_points=1024)
    test_dataset = PointCloudDataset(root_dir=data_dir, split='test', n_points=1024)

    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    points, labels = next(iter(loader))
    print(f"Batch shape: {points.shape}, Labels: {labels}")