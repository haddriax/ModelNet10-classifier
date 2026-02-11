"""Dataset returning sampled point clouds with automatic disk caching."""

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.builders.mesh_3D_builder import Mesh3DBuilder
from src.geometry import Mesh3D, Sampling
from src.dataset.base_modelnet_dataset import BaseModelNetDataset


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

        if cache_processed is None:
            cache_processed = (split == 'test')

        super().__init__(
            root_dir, split, use_existing_split,
            train_ratio, seed, cache_processed, verbose,
            transform,
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
        from tqdm import tqdm

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cached_data = []

        for idx, file_path in enumerate(tqdm(self.files, desc="Caching point clouds")):
            mesh: Mesh3D = Mesh3DBuilder.from_off_file(file_path)
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
