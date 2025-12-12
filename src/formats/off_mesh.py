import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from src.formats import format_parser
from src.geometry import points_cloud
from src.geometry.points_cloud import Sampling


class OffMesh:
    """
    3D mesh data from OFF (Object File Format) files.
    """

    regex_expr: re.Pattern = re.compile( r'_+\d+$')
    """ Regular expression pattern to extract the class from the file name. We set it for our usage with ModelNet10 """
    delimiter = ' '
    """ Delimiter used in the file to separate the entries on a same line, like coordinates """

    def __init__(self,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 name: str,
                 path: Path = None):
        self.path = path
        self.vertices = vertices
        self.faces = faces
        self.name = name
        self.point_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()

    def __str__(self):
        rep = f'OffObject: {self.name}, {len(self.vertices)} vertices, {len(self.faces)} faces\n'
        return rep

    @staticmethod
    def from_lines_list(lines: list[str],
                        name: str,
                        has_header: bool = True,
                        n_points: int = 2048,
                        sampling_method: points_cloud.Sampling=points_cloud.Sampling.UNIFORM)\
            -> 'OffMesh':
        """
        Creates the object from a list of lines representing edges (read from off files)
        :param lines:
        :param name:
        :param has_header:
        :param n_points: Number of points to keep in the sampled Points Cloud
        :param sampling_method: Specify how to sample the points from the mesh to the Points Cloud
        :return:
        """
        vertices: np.ndarray
        faces: np.ndarray
        vertices, faces = format_parser.parse_off(lines, has_header)

        # Sample points from mesh surface
        # Note: we were sampling from edges before, but it leads to some cluster of points so sampling from surface
        # is what we want to do.
        sampled_points = OffMesh.sample_points_from_mesh(vertices, faces, n_points, sampling_method)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(sampled_points)

        off_mesh = OffMesh(vertices, faces, name=name)
        off_mesh.point_cloud = pc
        return off_mesh

    @staticmethod
    def load_from_file(file_path: Path) -> 'OffMesh':
        # Extract the name of the class
        name = re.sub(OffMesh.regex_expr, '', file_path.stem)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OFF file not found: {file_path}")

        lines: list[str]
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid OFF file format in {file_path}: {e}")

        vertices, faces = format_parser.parse_off(lines, has_header=True)

        obj = OffMesh(vertices, faces, name=name)
        obj.path = file_path
        return obj

    @staticmethod
    def sample_points_from_mesh(vertices: np.ndarray,
                                faces: np.ndarray,
                                n_points: int = 2048,
                                sampling_method: Optional[Sampling] = Sampling.UNIFORM) -> np.ndarray:
        """Sample points from mesh surface and return a points cloud."""
        _mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
        _mesh.vertices = o3d.utility.Vector3dVector(vertices)
        _mesh.triangles = o3d.utility.Vector3iVector(faces)

        if sampling_method == Sampling.UNIFORM:
            pcd = _mesh.sample_points_uniformly(number_of_points=n_points)
        elif sampling_method == Sampling.POISSON:
            pcd = _mesh.sample_points_poisson_disk(number_of_points=n_points)
        elif sampling_method == Sampling.FARTHEST_POINT_SAMPLING:
            pcd = _mesh.sample_farthest_point_down_sample(num_samples=n_points)
        elif sampling_method == Sampling.RANDOM:
            pcd = _mesh.sample_points_uniformly(number_of_points=n_points)  # Use uniform as fallback
        else:
            raise ValueError(f"Unknown sampling method: {str(sampling_method)}")

        return np.asarray(pcd.points)

    def get_sampled_points(self, n_points: int = 2048,
                           sampling_method: Sampling = Sampling.UNIFORM) -> np.ndarray:
        """
        Sample points from mesh and return as a new numpy array.
        :param n_points: Number of points to keep in the sampled Points Cloud
        :param sampling_method: Specify how to sample the points from the mesh to the Points Cloud
        """
        return OffMesh.sample_points_from_mesh(
            self.vertices, self.faces, n_points, sampling_method
        )

    def sample_point_cloud(self, n_points: int = 2048,
                           sampling_method: Sampling = Sampling.UNIFORM) -> None:
        """
        Sample points from mesh and store in the point_cloud attribute.
        """
        sampled_points = OffMesh.sample_points_from_mesh(
            self.vertices, self.faces, n_points, sampling_method
        )
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(sampled_points)

# Test code to see if the basic sampling and display works
# Here we display the mesh as wireframe, with the cloud point version on top
# That's a good way to see how the sampling is performed especially if we want to avoid cluster of points
if __name__ == "__main__":
    test_path = Path("../night_stand_0001.off")
    off: OffMesh = OffMesh.load_from_file(test_path)


    # Create the mesh
    mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(off.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(off.faces)
    mesh.compute_vertex_normals()

    # Create wireframe from mesh edges
    wireframe: o3d.geometry.LineSet = o3d.geometry.LineSet.create_from_triangle_mesh(mesh=mesh)
    wireframe.paint_uniform_color([0.5, 0.5, 0.5])

    # Creation of the point cloud
    # Poisson disk sampling -> seems to do some clustering of points
    # vertex_poisson_disk vertex_poisson_disk = mesh.sample_points_poisson_disk(number_of_points=10000)
    # vertex_poisson_disk.paint_uniform_color([1, 0, 0])

    # Uniform sampling
    vertex_uniform = mesh.sample_points_uniformly(number_of_points=40000)
    vertex_uniform.paint_uniform_color([0, 1, 0])

    drawables = [
        wireframe,
        vertex_uniform
    ]

    o3d.visualization.draw_geometries(drawables)