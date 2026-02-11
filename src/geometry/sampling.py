"""Point cloud sampling methods."""

from enum import Enum


class Sampling(Enum):
    """Point cloud sampling methods.

    Defines strategies for sampling points from 3D mesh surfaces.
    Used by Mesh3D.sample_points() and throughout the training pipeline
    for configuring point cloud resolution.

    Values:
        UNIFORM: Uniformly random surface samples
        POISSON: Poisson disk sampling (evenly distributed)
        FARTHEST_POINT: Farthest point sampling (greedy, max coverage)
    """
    UNIFORM = "uniform"
    POISSON = "poisson"
    FARTHEST_POINT = "fps"
