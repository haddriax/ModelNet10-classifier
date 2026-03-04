"""Interactive OFF mesh inspector.

Opens a file-picker dialog rooted at the project's data/ folder, loads the
selected .off file, prints basic mesh stats, and displays the geometry in an
Open3D window (wireframe + point cloud for normal meshes; raw vertex cloud for
face-less files).

Usage::

    python -m scripts.view_mesh
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import numpy as np
import open3d as o3d

from src.builders.mesh_3D_builder import Mesh3DBuilder
from src.config import PROJECT_ROOT
from src.geometry.sampling import Sampling

_DATA_DIR = PROJECT_ROOT / "data"
_N_POINTS = 1024


def pick_off_file() -> Path | None:
    """Open a file-picker dialog and return the selected path, or None."""
    initial = _DATA_DIR if _DATA_DIR.exists() else Path.cwd()

    root = tk.Tk()
    root.withdraw()
    raw = filedialog.askopenfilename(
        title="Select an OFF mesh file",
        initialdir=str(initial),
        filetypes=[("OFF files", "*.off"), ("All files", "*.*")],
    )
    root.destroy()

    return Path(raw) if raw else None


def main() -> None:
    off_path = pick_off_file()
    if off_path is None:
        print("No file selected.")
        return

    mesh = Mesh3DBuilder.from_off_file(off_path)
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)

    print(f"File     : {off_path.name}")
    print(f"Vertices : {n_verts}")
    print(f"Faces    : {n_faces}")

    geometries: list = []

    if n_faces > 0:
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh.triangle_mesh)
        wireframe.paint_uniform_color([0.5, 0.5, 0.5])
        geometries.append(wireframe)
        points_np = mesh.sample_points(n_points=_N_POINTS, method=Sampling.UNIFORM)
    else:
        points_np = mesh.vertices.astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.paint_uniform_color([0.0, 0.8, 0.2])
    geometries.append(pcd)

    title = f"{off_path.name}  |  {n_verts} vertices  |  {n_faces} faces"
    o3d.visualization.draw_geometries(geometries, window_name=title, width=1280, height=800)


if __name__ == "__main__":
    main()
