"""Single-file inference script for ModelNet classifiers.

Loads one .off mesh, runs a forward pass through a trained checkpoint,
and prints the predicted class to the console.

Usage::

    python -m scripts.infer_single

Configure MODEL_PATH and OBJECT_PATH below before running.
"""

from pathlib import Path

import torch

from src.builders.mesh_3D_builder import Mesh3DBuilder
from src.deep_learning.inference import (
    detect_dataset_from_path,
    load_model_from_checkpoint,
    parse_checkpoint_config,
    run_inference,
)

# ---------------------------------------------------------------------------
# ← Configure these two paths before running
# ---------------------------------------------------------------------------
MODEL_PATH = Path("../models/sequential/modelnet40/2026-03-01_165413/PointNetPP_fps_pts1024_bs32_best.pth")
OBJECT_PATH = Path("../data/rebuilt/bathtub_0001(Clone).off")
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Parse model config from checkpoint filename
    parsed = parse_checkpoint_config(MODEL_PATH)
    if parsed is None:
        raise ValueError(
            f"Cannot parse model config from checkpoint filename: {MODEL_PATH.name}\n"
            f"Expected pattern: {{ModelName}}_{{sampling}}_pts{{N}}_bs{{B}}[_best].pth"
        )
    model_class, n_points, sampling = parsed

    # 2. Detect dataset and number of classes from checkpoint directory path
    detected = detect_dataset_from_path(MODEL_PATH)
    if detected is None:
        raise ValueError(
            f"Cannot detect dataset from checkpoint path: {MODEL_PATH}\n"
            f"Expected 'modelnet10' or 'modelnet40' somewhere in the path."
        )
    data_dir, num_classes = detected

    # 3. Build class-name mapping by scanning the data directory
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    idx_to_class = {i: name for i, name in enumerate(classes)}

    # 4. Load the trained model
    model = load_model_from_checkpoint(MODEL_PATH, model_class, num_classes, device)

    # 5. Load the .off mesh and sample a point cloud
    mesh = Mesh3DBuilder.from_off_file(OBJECT_PATH)
    points_np = mesh.sample_points(n_points=n_points, method=sampling)

    # 6. Run inference
    pred_idx, confidence = run_inference(model, points_np, device)
    pred_class = idx_to_class.get(pred_idx, f"<unknown index {pred_idx}>")

    # 7. Print result
    print(f"Object  : {OBJECT_PATH.name}")
    print(f"Model   : {MODEL_PATH.name}")
    print(f"→ Predicted class : {pred_class} (confidence: {confidence:.1%})")


if __name__ == "__main__":
    main()
