"""Sequential trainer for point cloud classifiers.

Trains each model with its own curated hyperparameters (sampling method, etc.)
rather than a full Cartesian grid search. One run per model, sequential execution.

Typical usage
-------------
Define a ``configs`` dict mapping model name → per-model options, then call
``run_sequential()``.  The only mandatory per-model key is ``"sampling"``.

Example::

    configs = {
        "PointNet":         {"sampling": "uniform"},
        "SimplePointNet":   {"sampling": "uniform"},
        "DGCNN":            {"sampling": "fps"},
        "PointNetPP":       {"sampling": "fps"},
        "PointTransformer": {"sampling": "fps"},
    }
    run_sequential(configs, n_points=1024, batch_size=32, epochs=50)

Entry point:
    python -m src.sequential_trainer
"""

import json
from datetime import datetime
from pathlib import Path

from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.dataset import PointCloudDataset
from src.deep_learning.model_trainer import ModelTrainer
from src.deep_learning.models import ALL_MODELS
from src.deep_learning.plotting import plot_sequential_results
from src.geometry import Sampling

# ---------------------------------------------------------------------------
# Sampling string → enum
# ---------------------------------------------------------------------------

SAMPLING_MAP: dict[str, Sampling] = {
    "uniform": Sampling.UNIFORM,
    "fps": Sampling.FARTHEST_POINT,
    "poisson": Sampling.POISSON,
}


# ---------------------------------------------------------------------------
# Dataset factory (mirrors train_classifier.make_datasets)
# ---------------------------------------------------------------------------

def _make_datasets(
    n_points: int,
    sampling_method: Sampling,
    data_dir: Path,
) -> tuple[PointCloudDataset, PointCloudDataset]:
    """Create a cached train/test PointCloudDataset pair."""
    train_ds = PointCloudDataset(
        root_dir=data_dir,
        split='train',
        n_points=n_points,
        sampling_method=sampling_method,
        use_existing_split=True,
        cache_processed=True,
    )
    test_ds = PointCloudDataset(
        root_dir=data_dir,
        split='test',
        n_points=n_points,
        sampling_method=sampling_method,
        use_existing_split=True,
        cache_processed=True,
    )
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_ES_METRICS: frozenset[str] = frozenset({"accuracy", "loss"})


def _resolve(key: str, model_cfg: dict, global_val):
    """Return the per-model override for *key* if present, else *global_val*."""
    return model_cfg.get(key, global_val)


def _validate_configs(configs: dict[str, dict]) -> None:
    """Raise early with clear messages if any config entry is invalid.

    Args:
        configs: Model-name → options dict, e.g. {"DGCNN": {"sampling": "fps"}}

    Raises:
        KeyError: If a model name is not found in ALL_MODELS.
        ValueError: If a sampling string is not in SAMPLING_MAP,
                    if the "sampling" key is missing,
                    or if any per-model ``lr``, ``patience``, or
                    ``early_stop_metric`` override is invalid.
    """
    for model_name, opts in configs.items():
        if model_name not in ALL_MODELS:
            available = ", ".join(sorted(ALL_MODELS.keys()))
            raise KeyError(
                f"Unknown model {model_name!r}. "
                f"Available models: {available}"
            )
        if "sampling" not in opts:
            raise ValueError(
                f"Config for {model_name!r} is missing the required "
                f"'sampling' key. Got: {opts}"
            )
        sampling_str = opts["sampling"]
        if sampling_str not in SAMPLING_MAP:
            available = ", ".join(sorted(SAMPLING_MAP.keys()))
            raise ValueError(
                f"Unknown sampling {sampling_str!r} for model {model_name!r}. "
                f"Available: {available}"
            )
        if "lr" in opts and not (
            isinstance(opts["lr"], (int, float)) and opts["lr"] > 0
        ):
            raise ValueError(
                f"lr must be a positive number for {model_name!r}, "
                f"got {opts['lr']!r}"
            )
        if "patience" in opts and not (
            isinstance(opts["patience"], int) and opts["patience"] >= 1
        ):
            raise ValueError(
                f"patience must be a positive integer for {model_name!r}, "
                f"got {opts['patience']!r}"
            )
        if (
            "early_stop_metric" in opts
            and opts["early_stop_metric"] not in _VALID_ES_METRICS
        ):
            raise ValueError(
                f"Invalid early_stop_metric {opts['early_stop_metric']!r} "
                f"for {model_name!r}. Choose from: {sorted(_VALID_ES_METRICS)}"
            )


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run_sequential(
    configs: dict[str, dict],
    *,
    n_points: int = 1024,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float | None = None,
    patience: int | None = None,
    early_stop_metric: str = "accuracy",
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> list[dict]:
    """Train each model in *configs* sequentially with its own sampling method.

    Args:
        configs: Mapping of model name → per-model options.
                 Required key per model: ``"sampling"`` (one of
                 ``"uniform"``, ``"fps"``, ``"poisson"``).
                 Optional per-model overrides: ``"lr"``, ``"patience"``,
                 ``"early_stop_metric"``.
        n_points: Number of points sampled per mesh (same for all models).
        batch_size: Training batch size (same for all models).
        epochs: Number of training epochs per model.
        lr: Global learning rate override.  ``None`` → auto ``0.001 * (bs/32)``.
            Can be overridden per-model via the config dict.
        patience: Global early-stopping patience (epochs without improvement).
                  ``None`` disables early stopping.  Can be overridden per-model.
        early_stop_metric: Metric watched globally by early stopping —
                           ``"accuracy"`` or ``"loss"``.  Can be overridden
                           per-model.
        data_dir: Root directory of ModelNet10 data.
        results_dir: Directory where JSON results and plots are saved.
        models_dir: Directory where model checkpoints are saved.

    Returns:
        List of run result dicts (one per model), each with keys:
        ``config``, ``run_name``, ``metrics``, ``timestamp``, ``status``
        (and ``error`` if the run failed).

    Raises:
        KeyError: On unknown model name.
        ValueError: On unknown sampling string, missing ``"sampling"`` key,
                    or invalid per-model ``lr`` / ``patience`` /
                    ``early_stop_metric`` value.
    """
    _validate_configs(configs)

    # Set up directories
    results_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "sequential").mkdir(parents=True, exist_ok=True)
    Path("runs").mkdir(parents=True, exist_ok=True)

    # Dataset cache keyed by (n_points, sampling_value) — avoids reloading
    # the same data for models that share sampling method and point count.
    dataset_cache: dict[tuple[int, str], tuple] = {}

    results: list[dict] = []
    start_time = datetime.now()
    total = len(configs)

    print(f"\n{'=' * 60}")
    print(f"SEQUENTIAL TRAINER: {total} model(s)")
    print(
        f"  n_points={n_points}  batch_size={batch_size}  epochs={epochs}  "
        f"lr={lr!r}  patience={patience!r}  metric={early_stop_metric!r}"
    )
    print(f"{'=' * 60}\n")

    for idx, (model_name, model_cfg) in enumerate(configs.items(), start=1):
        sampling_str = model_cfg["sampling"]
        sampling = SAMPLING_MAP[sampling_str]
        model_class = ALL_MODELS[model_name]

        # Resolve per-model overrides (fall back to global values)
        effective_lr       = _resolve("lr",                model_cfg, lr)
        effective_patience = _resolve("patience",          model_cfg, patience)
        effective_metric   = _resolve("early_stop_metric", model_cfg, early_stop_metric)

        run_name = (
            f"{model_name}_{sampling_str}"
            f"_pts{n_points}_bs{batch_size}"
        )
        save_path = models_dir / "sequential" / f"{run_name}.pth"
        experiment_name = f"sequential_{run_name}"

        print(f"[{idx}/{total}] Training: {model_name}")
        print(f"  Sampling : {sampling_str}")
        print(f"  n_points : {n_points}")
        print(f"  batch_sz : {batch_size}")
        print(f"  epochs   : {epochs}")
        print(f"  lr       : {effective_lr!r}")
        print(f"  patience : {effective_patience!r}")
        print(f"  es_metric: {effective_metric!r}")

        timestamp = datetime.now().isoformat()

        try:
            # Get or create datasets
            cache_key = (n_points, sampling_str)
            if cache_key not in dataset_cache:
                dataset_cache[cache_key] = _make_datasets(
                    n_points, sampling, data_dir
                )
            train_ds, test_ds = dataset_cache[cache_key]

            num_classes = len(train_ds.class_to_idx)
            model = model_class(num_classes=num_classes)

            trainer = ModelTrainer(
                train_dataset=train_ds,
                test_dataset=test_ds,
                model=model,
                save_model=save_path,
                batch_size=batch_size,
                experiment_name=experiment_name,
                lr=effective_lr,
                patience=effective_patience,
                early_stop_metric=effective_metric,
            )

            metrics = trainer.train(epochs=epochs, resume=False)

            print(
                f"  [OK] best_acc={metrics['best_test_acc']:.4f}  "
                f"F1={metrics['macro_f1']:.4f}\n"
            )

            results.append({
                "config": {
                    "model": model_name,
                    "sampling_method": sampling_str,
                    "n_points": n_points,
                    "batch_size": batch_size,
                    "epochs": epochs,
                },
                "run_name": run_name,
                "metrics": dict(metrics),
                "timestamp": timestamp,
                "status": "completed",
            })

        except Exception as exc:
            print(f"  [FAIL] {exc}\n")
            results.append({
                "config": {
                    "model": model_name,
                    "sampling_method": sampling_str,
                    "n_points": n_points,
                    "batch_size": batch_size,
                    "epochs": epochs,
                },
                "run_name": run_name,
                "metrics": None,
                "timestamp": timestamp,
                "status": "failed",
                "error": str(exc),
            })

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    end_time = datetime.now()
    duration_hours = (end_time - start_time).total_seconds() / 3600

    completed = [r for r in results if r["status"] == "completed"]
    best_run = None
    if completed:
        best = max(completed, key=lambda r: r["metrics"]["best_test_acc"])
        best_run = {
            "run_name": best["run_name"],
            "best_test_acc": best["metrics"]["best_test_acc"],
        }

    output = {
        "experiment_metadata": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_hours": round(duration_hours, 2),
            "num_models": total,
            "num_completed": len(completed),
            "n_points": n_points,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "early_stop_metric": early_stop_metric,
        },
        "runs": results,
        "best_run": best_run,
    }

    results_path = results_dir / "sequential_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved: {results_path}")

    # ------------------------------------------------------------------
    # Generate tailored plots
    # ------------------------------------------------------------------
    if completed:
        plot_sequential_results(results_path)
    else:
        print("No completed runs — skipping plots.")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"SEQUENTIAL TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Completed : {len(completed)}/{total}")
    if best_run:
        print(f"Best model: {best_run['run_name']}")
        print(f"Best acc  : {best_run['best_test_acc'] * 100:.2f}%")
    print(f"{'=' * 60}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configs = {
        "PointNet":         {"sampling": "uniform"},  # FPS adds cost with no benefit
        # "SimplePointNet":   {"sampling": "uniform"},  # same
        # "DGCNN":            {"sampling": "fps"},       # modest improvement
        "PointNetPP":       {"sampling": "fps"},       # non-negotiable
        "PointTransformer": {"sampling": "fps"},       # non-negotiable
    }

    run_sequential(
        configs,
        n_points=2048,
        batch_size=32,
        epochs=50,
    )
