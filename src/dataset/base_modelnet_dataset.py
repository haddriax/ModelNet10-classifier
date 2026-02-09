from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.builders.mesh_3D_builder import Mesh3DBuilder
from src.geometry.Mesh_3D import Mesh3D, Sampling


class BaseModelNetDataset(Dataset, ABC):
    """Abstract base dataset for ModelNet with caching support"""

    def __init__(
            self,
            root_dir: Path,
            split: str = 'train',
            use_existing_split: bool = True,
            train_ratio: float = 0.8,
            seed: int = 42,
            cache_processed: bool = False,
            verbose: bool = True
    ):
        if split not in ['train', 'test']:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

        self.root_dir = Path(root_dir)
        self.split = split
        self.use_existing_split = use_existing_split
        self.train_ratio = train_ratio
        self.seed = seed
        self.cache_processed = cache_processed
        self.verbose = verbose

        self.files: list[Path] = []  # Python 3.12 generic
        self.labels: list[int] = []
        self.class_to_idx: dict[str, int] = { }
        self.idx_to_class: dict[int, str] = { }
        self.cached_data: list | None = None  # Python 3.10+ union syntax

        self._build_index()
        if cache_processed:
            self._build_cache()

    def _build_index(self):
        """Build file index using appropriate split strategy"""
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])

        if not classes:
            raise FileNotFoundError(f"No class directories found in {self.root_dir}")

        # Build class mappings
        for class_name in classes:
            self.class_to_idx[class_name] = len(self.class_to_idx)
            self.idx_to_class[len(self.idx_to_class)] = class_name

        if self.use_existing_split:
            self._load_from_disk_split(classes)
        else:
            self._create_virtual_split(classes)

        if self.verbose:
            print(f"[{self.__class__.__name__}] {self.split}: {len(self.files)} samples "
                  f"from {len(classes)} classes")

    def _load_from_disk_split(self, classes: list[str]):
        """Load files from existing train/test folder structure"""
        for class_name in classes:
            class_split_dir = self.root_dir / class_name / self.split

            if not class_split_dir.exists():
                raise FileNotFoundError(
                    f"Split directory not found: {class_split_dir}. "
                    f"Set use_existing_split=False to create virtual splits."
                )

            class_files = sorted(list(class_split_dir.glob('*.off')))
            if not class_files and self.verbose:
                print(f"Warning: No .off files in {class_split_dir}")

            class_idx = self.class_to_idx[class_name]
            self.files.extend(class_files)
            self.labels.extend([class_idx] * len(class_files))

    def _create_virtual_split(self, classes: list[str]):
        """Create train/test split from all files, ignoring folder structure"""
        random.seed(self.seed)

        for class_name in classes:
            class_dir = self.root_dir / class_name

            # Collect all .off files
            all_class_files = []
            for split_folder in ['train', 'test']:
                split_dir = class_dir / split_folder
                if split_dir.exists():
                    all_class_files.extend(split_dir.glob('*.off'))

            if not all_class_files:
                all_class_files = list(class_dir.glob('*.off'))

            if not all_class_files:
                if self.verbose:
                    print(f"Warning: No .off files found for class {class_name}")
                continue

            all_class_files = sorted(all_class_files)
            random.shuffle(all_class_files)
            split_point = int(len(all_class_files) * self.train_ratio)

            class_files = (all_class_files[:split_point] if self.split == 'train'
                           else all_class_files[split_point:])

            class_idx = self.class_to_idx[class_name]
            self.files.extend(class_files)
            self.labels.extend([class_idx] * len(class_files))

    def _build_cache(self):
        """Build cache of processed data (override in subclass)"""
        pass

    def _get_cache_path(self) -> Path:
        """Get cache directory path (for subclass use)"""
        cache_name = f"{type(self).__name__}_{self.split}"
        return self.root_dir.parent / 'cache' / cache_name

    def __len__(self) -> int:
        return len(self.files)

    @abstractmethod
    def _process_mesh(self, mesh, idx: int) -> torch.Tensor:
        """Process mesh into desired representation"""
        pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:  # Python 3.12 generic
        """Get item with optional caching"""
        if self.cached_data is not None:
            data = self.cached_data[idx]
        else:
            try:
                mesh = Mesh3DBuilder.from_off_file(self.files[idx])
                data = self._process_mesh(mesh, idx)
            except Exception as e:
                raise RuntimeError(f"Failed to load {self.files[idx]}: {e}") from e

        return data, self.labels[idx]

    def get_class_name(self, idx: int) -> str:
        """Get class name from label index"""
        return self.idx_to_class[idx]


class PointCloudDataset(BaseModelNetDataset):
    """Dataset returning sampled point clouds with automatic disk caching.

    This dataset loads 3D meshes from ModelNet and samples point clouds from them.
    For reproducibility, the test set is cached by default (same point samples across runs).
    Training set uses dynamic sampling for data augmentation unless explicitly cached.

    Args:
        root_dir: Path to ModelNet directory containing class folders
        split: 'train' or 'test'
        n_points: Number of points to sample from each mesh
        sampling_method: Point sampling strategy (UNIFORM or FPS)
        use_existing_split: If True, uses train/test folders on disk.
                           If False, creates virtual split with train_ratio
        train_ratio: Training fraction when use_existing_split=False (default: 0.8)
        seed: Random seed for reproducible virtual splits
        cache_processed: If True, caches sampled point clouds to disk.
                        If None, auto-caches test set only (recommended)
        transform: Optional callable applied to point clouds (e.g., rotation, jitter)
        verbose: Print dataset loading information

    Attributes:
        cached_data: List of cached point cloud tensors if caching is enabled
        n_points: Number of points per sample
        sampling_method: Sampling strategy used

    Examples:
        >>> # Standard usage - test set cached, train set dynamic
        >>> train_ds = PointCloudDataset(data_dir, split='train', n_points=1024)
        >>> test_ds = PointCloudDataset(data_dir, split='test', n_points=1024)

        >>> # Custom virtual split
        >>> train_ds = PointCloudDataset(
        ...     data_dir, split='train', use_existing_split=False, train_ratio=0.9
        ... )

        >>> # Force cache training set (not recommended - loses augmentation)
        >>> train_ds = PointCloudDataset(
        ...     data_dir, split='train', cache_processed=True
        ... )

    Notes:
        - Cache is stored at: root_dir.parent/cache/pointcloud_{split}_{n_points}pts_{method}/
        - Cached files are named: {idx:05d}.npy
        - Cache is automatically loaded if it exists and matches dataset size
        - Test set caching ensures consistent evaluation across training runs
        - Training set dynamic sampling provides implicit data augmentation
    """

    def __init__(
            self,
            root_dir: Path,
            split: str = 'train',
            n_points: int = 1024,
            sampling_method: Sampling = Sampling.UNIFORM,
            use_existing_split: bool = True,
            train_ratio: float = 0.8,
            seed: int = 42,
            cache_processed: bool | None = None,
            transform: Callable | None = None,
            verbose: bool = True
    ):
        self.n_points = n_points
        self.sampling_method = sampling_method
        self.transform = transform

        if cache_processed is None:
            cache_processed = (split == 'test')

        super().__init__(
            root_dir, split, use_existing_split,
            train_ratio, seed, cache_processed, verbose,
        )

    def _get_cache_path(self) -> Path:
        """Cache path includes sampling parameters"""
        cache_name = f"pointcloud_{self.split}_{self.n_points}pts_{self.sampling_method.value}"
        return self.root_dir.parent / 'cache' / cache_name

    def _build_cache(self):
        """Build or load point cloud cache from disk"""
        cache_dir = self._get_cache_path()

        # Check if cache exists
        if cache_dir.exists() and len(list(cache_dir.glob('*.npy'))) == len(self.files):
            if self.verbose:
                print(f"Loading cached point clouds from {cache_dir}")
            self._load_cache_from_disk(cache_dir)
        else:
            if self.verbose:
                print(f"Creating point cloud cache at {cache_dir}")
            self._create_cache_on_disk(cache_dir)

    def _create_cache_on_disk(self, cache_dir: Path):
        """Sample and save point clouds to disk"""
        from src.builders.mesh_3D_builder import Mesh3DBuilder
        from tqdm import tqdm

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cached_data = []

        for idx, file_path in enumerate(tqdm(self.files, desc="Caching point clouds")):
            mesh = Mesh3DBuilder.from_off_file(file_path)
            points = mesh.sample_points(n_points=self.n_points, method=self.sampling_method, force_resample=True)

            # Save to disk
            cache_file = cache_dir / f'{idx:05d}.npy'
            np.save(cache_file, points)

            # Keep in memory
            self.cached_data.append(torch.from_numpy(points).float())

    def _load_cache_from_disk(self, cache_dir: Path):
        """Load cached point clouds from disk"""
        from tqdm import tqdm

        self.cached_data = []
        for idx in tqdm(range(len(self.files)), desc="Loading cache"):
            cache_file = cache_dir / f'{idx:05d}.npy'
            points = np.load(cache_file)
            self.cached_data.append(torch.from_numpy(points).float())

    def _process_mesh(self, mesh, idx: int) -> torch.Tensor:
        """Sample point cloud from mesh (when not using cache)"""
        points = mesh.sample_points(n_points=self.n_points, method=self.sampling_method)
        return torch.from_numpy(points).float()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get point cloud with optional transform"""
        data, label = super().__getitem__(idx)

        if self.transform:
            data = self.transform(data)

        return data, label


class MeshVerticesDataset(BaseModelNetDataset):
    """Dataset returning raw mesh vertices (padded/truncated)"""

    def __init__(self,
                 root_dir: Path,
                 split: str = 'train',
                 max_vertices: int = 2048,
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 transform: Optional[Callable] = None):
        self.max_vertices = max_vertices
        self.transform = transform
        super().__init__(root_dir, split, use_existing_split=True,
                         train_ratio=train_ratio, seed=seed)

    def _process_mesh(self, mesh: Mesh3D, idx: int) -> torch.Tensor:
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
                 transform: Optional[Callable] = None):
        self.n_points = n_points
        self.transform = transform
        super().__init__(root_dir, split, use_existing_split=True,
                         train_ratio=train_ratio, seed=seed)

    def _process_mesh(self, mesh: Mesh3D, idx: int) -> Dict[str, torch.Tensor]:
        points = mesh.sample_points(n_points=self.n_points)
        return {
            'point_cloud': torch.from_numpy(points).float(),
            'num_vertices': torch.tensor(len(mesh.vertices)),
            'num_faces': torch.tensor(len(mesh.faces)),
        }

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        mesh = Mesh3DBuilder.from_off_file(self.files[idx])
        data = self._process_mesh(mesh, idx)

        if self.transform:
            data = self.transform(data)

        return data, self.labels[idx]