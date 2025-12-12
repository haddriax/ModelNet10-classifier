from enum import Enum
from typing import Optional

import numpy as np
import open3d as o3d


class Sampling(Enum):
    """Point cloud sampling methods."""
    UNIFORM = "uniform"
    POISSON = "poisson"
    FARTHEST_POINT = "fps"

class Mesh3D:
    """3D mesh with optional point cloud sampling."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, name: str):
        """
        Initialize mesh.

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
            name: Mesh identifier (typically object class)
        """
        self.vertices: np.ndarray = vertices
        self.faces: np.ndarray = faces
        self.name: str = name

        # Build Open3D mesh
        self.triangle_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
        self.triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.triangle_mesh.triangles = o3d.utility.Vector3iVector(faces)
        self.triangle_mesh.compute_vertex_normals()

        # Lazy-loaded point cloud
        # We could use Open3D PointCloud, but for Machine Learning, numpy arrays are better
        self._point_cloud: Optional[np.ndarray] = None
        self._sampling_params: Optional[tuple] = None

    @property
    def mesh(self) -> o3d.geometry.TriangleMesh:
        """Get Open3D triangle mesh."""
        return self.triangle_mesh

    @property
    def point_cloud(self) -> Optional[np.ndarray]:
        """Get cached point cloud (None if not yet sampled)."""
        return self._point_cloud

    def sample_points(self,
                      n_points: int = 2048,
                      method: Sampling = Sampling.UNIFORM,
                      force_resample: bool = False) -> np.ndarray:
        """
        Sample points from mesh surface.

        Args:
            n_points: Number of points to sample
            method: Sampling strategy
            force_resample: Recompute even if cached

        Returns:
            Nx3 array of sampled points
        """
        params = (n_points, method)

        # Return cached if available and params match
        if not force_resample and self._point_cloud is not None and self._sampling_params == params:
            return self._point_cloud

        # Sample based on method
        if method == Sampling.UNIFORM:
            pcd = self.triangle_mesh.sample_points_uniformly(n_points)
        elif method == Sampling.POISSON:
            pcd = self.triangle_mesh.sample_points_poisson_disk(n_points)
        elif method == Sampling.FARTHEST_POINT:
            pcd = self.triangle_mesh.sample_points_uniformly(n_points * 2)  # Oversample first
            pcd = pcd.farthest_point_down_sample(n_points)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Cache result
        self._point_cloud = np.asarray(pcd.points, dtype=np.float32)
        self._sampling_params = params

        return self._point_cloud

    def __str__(self) -> str:
        return f"Mesh3D(name='{self.name}', vertices={len(self.vertices)}, faces={len(self.faces)})"

    def __repr__(self) -> str:
        return self.__str__()
