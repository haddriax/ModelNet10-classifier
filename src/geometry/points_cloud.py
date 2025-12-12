from enum import Enum
from typing import Optional

import numpy as np
import open3d as o3d

class Sampling(Enum):
    """
    Points Cloud sampling method selection.
    """
    NONE='none'
    RANDOM = 'random'
    UNIFORM = 'uniform'
    FARTHEST_POINT_SAMPLING = 'fps'
    POISSON = 'poisson'

def point_cloud_from_vertices(
        vertices: np.ndarray,
        sample: Optional[Sampling] = Sampling.UNIFORM
) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)

    if sample == Sampling.RANDOM:
        point_cloud = point_cloud.random_down_sample(sampling_ratio=0.5)
    elif sample == Sampling.UNIFORM:
        point_cloud = point_cloud.uniform_down_sample(every_k_points=2)
    elif sample == Sampling.FARTHEST_POINT_SAMPLING:
        size = round(len(point_cloud.points) / 2)
        point_cloud = point_cloud.farthest_point_down_sample(num_samples=size)
    elif sample == Sampling.NONE:
        pass

    return point_cloud


