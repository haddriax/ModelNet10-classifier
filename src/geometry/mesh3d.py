"""3D mesh representation with point cloud sampling."""

import warnings

import numpy as np
import open3d as o3d

from src.geometry.sampling import Sampling


def _reconstruct_faces_poisson(
    vertices: np.ndarray,
    name: str,
    depth: int = 7,
    density_quantile_threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct triangle faces from raw scan vertices via Poisson surface reconstruction.

    Voxel-downsamples the input first to collapse overlapping scan passes, then
    runs Poisson reconstruction and prunes low-density boundary artifacts.

    Parameters are tuned aggressively by default because this function is used on imperfect scans.

    Args:
        vertices:                   Nx3 float array of scan points.
        name:                       Mesh name used in warning messages.
        depth:                      Octree depth controlling output resolution.
                                    depth=6 → ~4k faces, depth=8 → ~65k faces.
        density_quantile_threshold: Bottom quantile of Poisson density to prune.
                                    Removes extrapolation artifacts at scan boundaries.
                                    0.05 = remove lowest 5% density vertices.

    Returns:
        Tuple of (vertices, faces) as float32/int32 arrays.
        Returns the original vertices with an empty face array if reconstruction fails.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Collapse overlapping scan passes into a single surface layer.
    # voxel_size is set to 2× the 90th-percentile nearest-neighbour distance
    # to aggressively merge nearby duplicate points without destroying geometry.
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    voxel_size = float(np.percentile(distances, 90)) * 2.0
    pcd = pcd.voxel_down_sample(voxel_size)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Prune Poisson extrapolation artifacts at scan boundaries.
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile_threshold))

    faces = np.asarray(mesh.triangles, dtype=np.int32)
    reconstructed_vertices = np.asarray(mesh.vertices, dtype=np.float32)

    if faces.shape[0] == 0:
        warnings.warn(
            f"Mesh '{name}': Poisson reconstruction produced no triangles. "
            "Point cloud may be too sparse or degenerate. "
            "Falling back to vertex-based sampling.",
            UserWarning,
            stacklevel=3,
        )
        return vertices, np.empty((0, 3), dtype=np.int32)

    return reconstructed_vertices, faces


class Mesh3D:
    """3D mesh with lazy point cloud sampling.

    Wraps vertex/face data with an Open3D TriangleMesh for sampling and
    visualization. When no faces are provided (e.g. raw scan data), faces are
    automatically reconstructed via Poisson surface reconstruction. Sampled
    point clouds are cached in-memory.

    Args:
        vertices: Nx3 array of vertex coordinates.
        faces:    Mx3 array of triangular face indices. Pass an empty (0, 3)
                  array to trigger automatic Poisson face reconstruction.
        name:     Mesh identifier (typically the object class label).

    Raises:
        ValueError: If vertices/faces have invalid shape, contain NaN,
                    or face indices are out of range.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, name: str) -> None:
        self._validate(vertices, faces)
        self.name = name

        if faces.shape[0] == 0:
            vertices, faces = _reconstruct_faces_poisson(vertices, name)

        self.vertices: np.ndarray = vertices
        self.faces: np.ndarray = faces

        self.triangle_mesh = o3d.geometry.TriangleMesh()
        self.triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.triangle_mesh.triangles = o3d.utility.Vector3iVector(faces)
        self.triangle_mesh.compute_vertex_normals()

        self._point_cloud: np.ndarray | None = None
        self._sampling_params: tuple | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(vertices: np.ndarray, faces: np.ndarray) -> None:
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be Nx3, got shape {vertices.shape}")
        if vertices.shape[0] == 0:
            raise ValueError("vertices array must not be empty")
        if np.any(np.isnan(vertices)):
            raise ValueError("vertices contain NaN values")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be Mx3, got shape {faces.shape}")
        if faces.shape[0] > 0 and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError(
                f"face indices must be in [0, {len(vertices)}), "
                f"got range [{faces.min()}, {faces.max()}]"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mesh(self) -> o3d.geometry.TriangleMesh:
        """Open3D triangle mesh."""
        return self.triangle_mesh

    @property
    def point_cloud(self) -> np.ndarray | None:
        """Cached point cloud, or None if not yet sampled."""
        return self._point_cloud

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_points(
        self,
        n_points: int = 2048,
        method: Sampling = Sampling.UNIFORM,
        force_resample: bool = False,
    ) -> np.ndarray:
        """Sample points from the mesh surface.

        Args:
            n_points:       Number of points to sample.
            method:         Sampling strategy.
            force_resample: Bypass cache and recompute.

        Returns:
            Nx3 float32 array of sampled points.
        """
        params = (n_points, method)

        if not force_resample and self._point_cloud is not None and self._sampling_params == params:
            return self._point_cloud

        if len(self.faces) == 0:
            self._point_cloud = self._sample_from_vertices(n_points, method)
            self._sampling_params = params
            return self._point_cloud

        if method == Sampling.UNIFORM:
            pcd = self.triangle_mesh.sample_points_uniformly(n_points)
        elif method == Sampling.POISSON:
            pcd = self.triangle_mesh.sample_points_poisson_disk(n_points)
        elif method == Sampling.FARTHEST_POINT:
            pcd = self.triangle_mesh.sample_points_uniformly(n_points * 2)
            pcd = pcd.farthest_point_down_sample(n_points)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        self._point_cloud = np.asarray(pcd.points, dtype=np.float32)
        self._sampling_params = params
        return self._point_cloud

    def _sample_from_vertices(self, n_points: int, method: Sampling) -> np.ndarray:
        """Vertex-based fallback when no face data is available after reconstruction.

        Args:
            n_points: Number of points to return.
            method:   Sampling strategy.

        Returns:
            Nx3 float32 array of sampled vertex positions.
        """
        warnings.warn(
            f"Mesh '{self.name}' has no faces — falling back to vertex sampling. "
            "Results may differ from surface sampling.",
            UserWarning,
            stacklevel=3,
        )
        num_v = len(self.vertices)
        if method == Sampling.FARTHEST_POINT and num_v > n_points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.vertices)
            pcd = pcd.farthest_point_down_sample(n_points)
            return np.asarray(pcd.points, dtype=np.float32)

        indices = np.random.choice(num_v, n_points, replace=num_v < n_points)
        return self.vertices[indices].astype(np.float32)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"Mesh3D(name='{self.name}', "
            f"vertices={len(self.vertices)}, "
            f"faces={len(self.faces)})"
        )

    def __repr__(self) -> str:
        return self.__str__()