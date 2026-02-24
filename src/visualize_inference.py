"""Interactive 3D inference visualizer for ModelNet10 classifiers.

Loads a trained checkpoint from an interactive terminal menu, then displays
random test samples in Open3D (wireframe + point cloud) while running live
inference. Press SPACE for the next sample, Q to quit.

Entry point:
    python -m src.visualize_inference
"""

import random
import re
import sys
from pathlib import Path

import open3d as o3d
import torch
import torch.nn.functional as F

from src.builders.mesh_3D_builder import Mesh3DBuilder
from src.config import DATA_DIR, MODELS_DIR
from src.dataset.point_cloud_dataset import PointCloudDataset
from src.deep_learning.models import ALL_MODELS
from src.geometry import Sampling

# ---------------------------------------------------------------------------
# Checkpoint scanning and selection
# ---------------------------------------------------------------------------

SAMPLING_MAP: dict[str, Sampling] = {
    "uniform": Sampling.UNIFORM,
    "fps": Sampling.FARTHEST_POINT,
    "poisson": Sampling.POISSON,
}

# Matches: {ModelName}_{sampling}_pts{N}_bs{B}[_best].pth
_CKPT_PATTERN = re.compile(
    r'^([A-Za-z]+(?:PP)?)_(uniform|fps|poisson)_pts(\d+)_bs\d+',
    re.IGNORECASE,
)


def scan_checkpoints(models_dir: Path) -> list[Path]:
    """Return all .pth files found under models_dir, sorted."""
    checkpoints = sorted(models_dir.rglob("*.pth"))
    if not checkpoints:
        print(f"No .pth checkpoints found under: {models_dir}")
        print("Run `make train` first to produce trained checkpoints.")
        sys.exit(1)
    return checkpoints


def parse_checkpoint_config(
    path: Path,
) -> tuple[type, int, Sampling] | None:
    """Parse model class, n_points and sampling method from a checkpoint filename.

    Args:
        path: Path to the .pth checkpoint file.

    Returns:
        (model_class, n_points, sampling) if filename matches convention, else None.
    """
    match = _CKPT_PATTERN.match(path.stem)
    if not match:
        return None

    model_name, sampling_str, n_points_str = match.groups()
    model_class = ALL_MODELS.get(model_name)
    if model_class is None:
        return None

    sampling = SAMPLING_MAP.get(sampling_str.lower())
    if sampling is None:
        return None

    return model_class, int(n_points_str), sampling


def interactive_menu(checkpoints: list[Path]) -> Path:
    """Print a numbered checkpoint list and let the user pick one.

    Args:
        checkpoints: List of available .pth file paths.

    Returns:
        The selected checkpoint path.
    """
    print("\nAvailable checkpoints:")
    for i, ckpt in enumerate(checkpoints, start=1):
        # Show path relative to project root for readability
        try:
            rel = ckpt.relative_to(Path.cwd())
        except ValueError:
            rel = ckpt
        print(f"  [{i}] {rel}")

    while True:
        raw = input(f"\nSelect checkpoint (1-{len(checkpoints)}): ").strip()
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(checkpoints):
                return checkpoints[choice - 1]
        print(f"  Please enter a number between 1 and {len(checkpoints)}.")


def resolve_config_interactively(
    path: Path,
) -> tuple[type, int, Sampling]:
    """Try to parse config from filename; fall back to manual input if needed.

    Args:
        path: Checkpoint file path.

    Returns:
        (model_class, n_points, sampling)
    """
    parsed = parse_checkpoint_config(path)
    if parsed is not None:
        return parsed

    # --- Fallback: ask user ---
    print(f"\nCould not parse config from filename: {path.name}")
    print(f"Available models: {', '.join(ALL_MODELS.keys())}")
    while True:
        name = input("Model name: ").strip()
        if name in ALL_MODELS:
            model_class = ALL_MODELS[name]
            break
        print(f"  Unknown model. Choose from: {', '.join(ALL_MODELS.keys())}")

    while True:
        raw = input("n_points (e.g. 256 / 512 / 1024): ").strip()
        if raw.isdigit() and int(raw) > 0:
            n_points = int(raw)
            break
        print("  Please enter a positive integer.")

    print(f"Sampling methods: {', '.join(SAMPLING_MAP.keys())}")
    while True:
        raw = input("Sampling method: ").strip().lower()
        if raw in SAMPLING_MAP:
            sampling = SAMPLING_MAP[raw]
            break
        print(f"  Choose from: {', '.join(SAMPLING_MAP.keys())}")

    return model_class, n_points, sampling


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    path: Path,
    model_class: type,
    num_classes: int = 10,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Instantiate model and load weights from checkpoint.

    Args:
        path: Path to .pth checkpoint file.
        model_class: The nn.Module subclass to instantiate.
        num_classes: Number of output classes.
        device: Target device (defaults to CUDA if available).

    Returns:
        Model in eval mode on the specified device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class(num_classes=num_classes)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)


def run_inference(
    model: torch.nn.Module,
    points_np,
    device: torch.device,
) -> tuple[int, float]:
    """Run a single forward pass and return (predicted_class_idx, confidence).

    Args:
        model: Trained model in eval mode.
        points_np: numpy array [N, 3] of point cloud coordinates.
        device: Device for inference.

    Returns:
        (predicted_class_index, confidence_score in [0, 1])
    """
    x = torch.from_numpy(points_np).float().unsqueeze(0).to(device)  # [1, N, 3]
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred_idx = int(probs.argmax(1).item())
    confidence = float(probs[0, pred_idx].item())
    return pred_idx, confidence


# ---------------------------------------------------------------------------
# Open3D geometry helpers
# ---------------------------------------------------------------------------

def build_geometries(
    off_path: Path,
    n_points: int,
    sampling: Sampling,
):
    """Load mesh and build Open3D geometries for display.

    Args:
        off_path: Path to the .off mesh file.
        n_points: Number of points to sample.
        sampling: Sampling method enum value.

    Returns:
        (wireframe LineSet, point cloud PointCloud, points numpy [N, 3])
    """
    mesh = Mesh3DBuilder.from_off_file(off_path)

    # Gray wireframe
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(
        mesh.triangle_mesh
    )
    wireframe.paint_uniform_color([0.5, 0.5, 0.5])

    # Green point cloud (same points the model will see)
    points_np = mesh.sample_points(n_points=n_points, method=sampling)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.paint_uniform_color([0.0, 0.8, 0.2])

    return wireframe, pcd, points_np


# ---------------------------------------------------------------------------
# Main visualizer loop
# ---------------------------------------------------------------------------

def run_visualizer(
    model: torch.nn.Module,
    dataset: PointCloudDataset,
    n_points: int,
    sampling: Sampling,
    model_name: str,
    device: torch.device,
) -> None:
    """Open the Open3D window and run the interactive inference loop.

    Args:
        model: Trained model in eval mode.
        dataset: Test PointCloudDataset instance.
        n_points: Point count to sample per mesh.
        sampling: Sampling method to use.
        model_name: Display name for the window title.
        device: Inference device.
    """
    # Shuffle dataset indices for random traversal order
    random_order = list(range(len(dataset)))
    random.shuffle(random_order)

    # Shared mutable state (closure-friendly dict)
    state = {"pos": 0}

    def load_sample(vis) -> bool:
        """Load the current sample, run inference, update display."""
        idx = random_order[state["pos"]]
        off_path = dataset.files[idx]
        true_label_idx = dataset.labels[idx]
        true_label = dataset.get_class_name(true_label_idx)

        wireframe, pcd, points_np = build_geometries(off_path, n_points, sampling)
        pred_idx, confidence = run_inference(model, points_np, device)
        pred_label = dataset.get_class_name(pred_idx)

        correct_mark = "✓" if pred_label == true_label else "✗"

        # Update 3D view
        vis.clear_geometries()
        vis.add_geometry(wireframe)
        vis.add_geometry(pcd)
        vis.reset_view_point(True)

        # Terminal output
        sep = "━" * 57
        print(f"\n{sep}")
        print(
            f"Sample {state['pos'] + 1}/{len(random_order)}"
            f"  |  {off_path.name}"
        )
        print(f"Ground truth : {true_label}")
        print(
            f"Predicted    : {pred_label}  {correct_mark}"
            f"  (confidence: {confidence * 100:.1f}%)"
        )
        print(sep)
        print("Press SPACE for next sample  |  Q to quit")

        return False

    def on_spacebar(vis) -> bool:
        state["pos"] = (state["pos"] + 1) % len(random_order)
        load_sample(vis)
        return False

    def on_quit(vis) -> bool:
        vis.close()
        return False

    # Build window (title is static — Open3D has no runtime title-change API)
    window_title = f"Inference  —  {model_name}  |  {n_points} pts  |  {sampling.value}"
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_title, width=1280, height=800)

    # Register keyboard callbacks
    vis.register_key_callback(32, on_spacebar)          # SPACE
    vis.register_key_callback(ord("Q"), on_quit)        # Q
    vis.register_key_callback(256, on_quit)             # Escape (GLFW_KEY_ESCAPE)

    # Load the first sample
    load_sample(vis)

    print(f"\nOpen3D window open: {window_title!r}")
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Discover checkpoints
    checkpoints = scan_checkpoints(MODELS_DIR)

    # 2. User picks one
    selected = interactive_menu(checkpoints)

    # 3. Parse or ask for config
    model_class, n_points, sampling = resolve_config_interactively(selected)
    model_name = model_class.__name__
    print(
        f"\nLoaded: {model_name}  |  {sampling.value}  |  {n_points} pts"
        f"\nCheckpoint: {selected}"
    )

    # 4. Load model weights
    model = load_model_from_checkpoint(selected, model_class, device=device)

    # 5. Build test dataset (cache_processed=True re-uses cached test points)
    dataset = PointCloudDataset(
        root_dir=DATA_DIR,
        split='test',
        n_points=n_points,
        sampling_method=sampling,
        use_existing_split=True,
        cache_processed=True,
    )

    if len(dataset) == 0:
        print("Test dataset is empty — check DATA_DIR and split folders.")
        sys.exit(1)

    print(f"\nPress SPACE to cycle through random test samples | Q to quit\n")

    # 6. Launch interactive visualizer
    run_visualizer(model, dataset, n_points, sampling, model_name, device)


if __name__ == "__main__":
    main()
