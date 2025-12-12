from src.geometry.Mesh_3D import Mesh3D
import open3d as o3d

if __name__ == "__main__":
    from pathlib import Path
    from src.builders.mesh_3D_builder import Mesh3DBuilder

    mesh: Mesh3D = Mesh3DBuilder.from_off_file(Path("./night_stand_0001.off"))

    points = mesh.sample_points(n_points=4096)

    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh.triangle_mesh)
    wireframe.paint_uniform_color([0.5, 0.5, 0.5])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([wireframe, pcd])