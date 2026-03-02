"""Shared dataset factory for ModelNet10 point-cloud training pipelines.

Both the sequential trainer (:mod:`src.deep_learning.sequential_trainer`)
and the grid-search entry point (:mod:`src.train_classifier`) use
:func:`make_datasets` to produce train/test dataset pairs.

The :data:`DatasetFactory` type alias is the expected signature for any
callable passed to :class:`~src.deep_learning.grid_search.GridSearch`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from src.config import DATA_DIR
from src.dataset import PointCloudDataset
from src.geometry import Sampling

# Type alias: factory that creates (train_dataset, test_dataset) for given params.
# Matches the signature used by GridSearch and run_sequential.
DatasetFactory = Callable[[int, Sampling], tuple[Dataset, Dataset]]


def make_datasets(
    n_points: int,
    sampling_method: Sampling,
    data_dir: Path = DATA_DIR,
) -> tuple[PointCloudDataset, PointCloudDataset]:
    """Create a cached train/test :class:`~src.dataset.PointCloudDataset` pair.

    The test split is cached to disk (keyed by *n_points* + *sampling_method*);
    the training split is re-sampled dynamically each epoch for augmentation.

    Args:
        n_points: Number of points sampled per mesh.
        sampling_method: Point-cloud sampling strategy
                         (:attr:`~src.geometry.Sampling.UNIFORM`,
                         :attr:`~src.geometry.Sampling.FARTHEST_POINT`, or
                         :attr:`~src.geometry.Sampling.POISSON`).
        data_dir: Root directory of the ModelNet10 dataset.

    Returns:
        ``(train_dataset, test_dataset)`` tuple ready for use with
        :class:`~src.deep_learning.model_trainer.ModelTrainer`.
    """
    train_ds = PointCloudDataset(
        root_dir=data_dir,
        split='train',
        n_points=n_points,
        sampling_method=sampling_method,
        use_existing_split=True,
        cache_processed=True,
    )
    test_ds = PointCloudDataset(
        root_dir=data_dir,
        split='test',
        n_points=n_points,
        sampling_method=sampling_method,
        use_existing_split=True,
        cache_processed=True,
    )
    return train_ds, test_ds
