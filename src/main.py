import random

from src.geometry import Mesh3D
import open3d as o3d
from src.builders.mesh_3D_builder import Mesh3DBuilder
from src.config import DATA_DIR

if __name__ == "__main__":
    model_files = list(DATA_DIR.rglob("*.off"))
    random.shuffle(model_files)

    current_idx = 0


    def load_model(idx):
        """Load and return geometries for model at index"""
        mesh: Mesh3D = Mesh3DBuilder.from_off_file(model_files[idx])
        points = mesh.sample_points(n_points=4096)

        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh.triangle_mesh)
        wireframe.paint_uniform_color([0.5, 0.5, 0.5])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0, 1, 0])

        return wireframe, pcd


    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    wireframe, pcd = load_model(current_idx)
    vis.add_geometry(wireframe)
    vis.add_geometry(pcd)


    def change_model(vis, direction):
        global current_idx
        current_idx = (current_idx + direction) % len(model_files)

        vis.clear_geometries()
        wireframe, pcd = load_model(current_idx)
        vis.add_geometry(wireframe)
        vis.add_geometry(pcd)
        print(f"Loaded: {model_files[current_idx].name}")
        return False

    vis.register_key_callback(ord("N"), lambda vis: change_model(vis, 1))
    vis.register_key_callback(ord("P"), lambda vis: change_model(vis, -1))
    vis.register_key_callback(262, lambda vis: change_model(vis, 1))
    vis.register_key_callback(263, lambda vis: change_model(vis, -1))

    print(f"Loaded: {model_files[current_idx].name}")
    print("Controls: N/Right Arrow = Next, P/Left Arrow = Previous")

    vis.run()
    vis.destroy_window()
