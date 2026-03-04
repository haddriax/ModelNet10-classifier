"""Inference utilities for checkpoint loading and forward passes.

Shared between :mod:`scripts.visualize_inference` and :mod:`scripts.infer_single`.
Provides helpers for resolving model config from checkpoint filenames, loading
weights, and running single-sample inference.
"""

import re
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import DATA_DIR, MODELNET40_DIR
from src.deep_learning.models import ALL_MODELS
from src.geometry import Sampling

# ---------------------------------------------------------------------------
# Checkpoint parsing constants
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

_DATASET_MAP: dict[str, tuple[Path, int]] = {
    "modelnet10": (DATA_DIR,       10),
    "modelnet40": (MODELNET40_DIR, 40),
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_dataset_from_path(path: Path) -> tuple[Path, int] | None:
    """Infer data_dir and num_classes from the checkpoint's directory tree.

    Looks for 'modelnet10' or 'modelnet40' among the path components.

    Args:
        path: Path to the checkpoint file.

    Returns:
        (data_dir, num_classes) if a known dataset name is found, else None.
    """
    for part in path.parts:
        key = part.lower()
        if key in _DATASET_MAP:
            return _DATASET_MAP[key]
    return None


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


def load_model_from_checkpoint(
    path: Path,
    model_class: type,
    num_classes: int,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Instantiate model and load weights from checkpoint.

    Args:
        path: Path to .pth checkpoint file.
        model_class: The nn.Module subclass to instantiate.
        num_classes: Number of output classes (10 for ModelNet10, 40 for ModelNet40).
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
    x = torch.from_numpy(points_np).float()

    # Unit-sphere normalisation — must match PointCloudDataset._normalize_point_cloud
    centroid = x.mean(dim=0)
    x = x - centroid
    max_dist = x.norm(dim=1).max()
    if max_dist > 0:
        x = x / max_dist

    x = x.unsqueeze(0).to(device)  # [1, N, 3]
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred_idx = int(probs.argmax(1).item())
    confidence = float(probs[0, pred_idx].item())
    return pred_idx, confidence
