"""Abstract base dataset for ModelNet with caching and split management."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.builders.mesh_3D_builder import Mesh3DBuilder


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
            verbose: bool = True,
            transform: Callable | None = None
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
        self.transform = transform

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
        """Get item with optional caching and transform"""
        if self.cached_data is not None:
            data = self.cached_data[idx]
        else:
            try:
                mesh = Mesh3DBuilder.from_off_file(self.files[idx])
                data = self._process_mesh(mesh, idx)
            except Exception as e:
                raise RuntimeError(f"Failed to load {self.files[idx]}: {e}") from e

        if self.transform:
            data = self.transform(data)

        return data, self.labels[idx]

    def get_class_name(self, idx: int) -> str:
        """Get class name from label index"""
        return self.idx_to_class[idx]