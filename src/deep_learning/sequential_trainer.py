"""Sequential trainer for point cloud classifiers — library module.

Trains each model with its own curated hyperparameters (sampling method, etc.)
rather than a full Cartesian grid search. One run per model, sequential execution.

Typical usage
-------------
Define a ``configs`` dict mapping model name → :class:`~src.deep_learning.configs.ModelConfig`,
then call :func:`run_sequential`.  The only mandatory per-model field is ``sampling``.

Example::

    from src.deep_learning.configs import ModelConfig
    from src.deep_learning.sequential_trainer import run_sequential

    configs = {
        "PointNet":         ModelConfig(sampling="uniform"),
        "SimplePointNet":   ModelConfig(sampling="uniform"),
        "DGCNN":            ModelConfig(sampling="fps"),
        "PointNetPP":       ModelConfig(sampling="fps", epochs=100),
        "PointTransformer": ModelConfig(sampling="fps"),
    }
    run_sequential(configs, n_points=1024, batch_size=32, epochs=50)

Entry point:
    python -m src.sequential_trainer   (canonical runner — edit configs there)
"""

from datetime import datetime
from pathlib import Path

from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.deep_learning.configs import ModelConfig
from src.deep_learning.dataset_factory import make_datasets
from src.deep_learning.model_trainer import ModelTrainer
from src.deep_learning.models import ALL_MODELS
from src.deep_learning.plotting import plot_sequential_results
from src.deep_learning.result_utils import find_best_run, save_json
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
# Private helpers
# ---------------------------------------------------------------------------

def _validate(configs: dict[str, ModelConfig]) -> None:
    """Raise early if any model name or config value is invalid.

    Args:
        configs: Mapping of model name → :class:`ModelConfig`.

    Raises:
        TypeError: If any value is not a :class:`ModelConfig` instance.
        KeyError: If any model name is not found in :data:`ALL_MODELS`.
    """
    for model_name, model_cfg in configs.items():
        if not isinstance(model_cfg, ModelConfig):
            raise TypeError(
                f"Expected ModelConfig for {model_name!r}, "
                f"got {type(model_cfg).__name__!r}. "
                f"Use: ModelConfig(sampling='fps')"
            )
        if model_name not in ALL_MODELS:
            available = ", ".join(sorted(ALL_MODELS.keys()))
            raise KeyError(
                f"Unknown model {model_name!r}. "
                f"Available models: {available}"
            )


def _run_one(
    model_name: str,
    model_cfg: ModelConfig,
    *,
    n_points: int,
    batch_size: int,
    effective_lr: float | None,
    effective_patience: int | None,
    effective_metric: str,
    effective_epochs: int,
    dataset_cache: dict,
    models_dir: Path,
    data_dir: Path,
    idx: int,
    total: int,
) -> dict:
    """Train one model and return its result dict.

    Args:
        model_name: Key into :data:`ALL_MODELS`.
        model_cfg: Per-model configuration (sampling already resolved).
        n_points: Number of points per mesh.
        batch_size: Training batch size.
        effective_lr: Learning rate (global fallback already applied).
        effective_patience: Early-stopping patience (``None`` = disabled).
        effective_metric: Early-stopping metric name.
        effective_epochs: Number of training epochs for this run.
        dataset_cache: Shared cache dict keyed by ``(n_points, sampling_str)``.
        models_dir: Root directory for model checkpoints.
        data_dir: Root directory of ModelNet10 data.
        idx: 1-based index of this run (for progress printing).
        total: Total number of runs (for progress printing).

    Returns:
        Result dict with keys ``config``, ``run_name``, ``metrics``,
        ``timestamp``, ``status`` (and ``error`` on failure).
    """
    sampling_str = model_cfg.sampling
    sampling = SAMPLING_MAP[sampling_str]
    model_class = ALL_MODELS[model_name]

    run_name = f"{model_name}_{sampling_str}_pts{n_points}_bs{batch_size}"
    save_path = models_dir / "sequential" / f"{run_name}.pth"
    experiment_name = f"sequential_{run_name}"

    print(f"\n[{idx}/{total}] Training: {model_name}")
    print(f"  Sampling : {sampling_str}")
    print(f"  n_points : {n_points}")
    print(f"  batch_sz : {batch_size}")
    print(f"  epochs   : {effective_epochs}")
    print(f"  lr       : {effective_lr!r}")
    print(f"  patience : {effective_patience!r}")
    print(f"  es_metric: {effective_metric!r}")

    timestamp = datetime.now().isoformat()

    try:
        cache_key = (n_points, sampling_str)
        if cache_key not in dataset_cache:
            dataset_cache[cache_key] = make_datasets(n_points, sampling, data_dir)
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

        metrics = trainer.train(epochs=effective_epochs, resume=False)

        print(
            f"  [OK] best_acc={metrics['best_test_acc']:.4f}  "
            f"F1={metrics['macro_f1']:.4f}\n"
        )

        return {
            "config": {
                "model": model_name,
                "sampling_method": sampling_str,
                "n_points": n_points,
                "batch_size": batch_size,
                "epochs": effective_epochs,
            },
            "run_name": run_name,
            "metrics": dict(metrics),
            "timestamp": timestamp,
            "status": "completed",
        }

    except Exception as exc:
        print(f"  [FAIL] {exc}\n")
        return {
            "config": {
                "model": model_name,
                "sampling_method": sampling_str,
                "n_points": n_points,
                "batch_size": batch_size,
                "epochs": effective_epochs,
            },
            "run_name": run_name,
            "metrics": None,
            "timestamp": timestamp,
            "status": "failed",
            "error": str(exc),
        }


def _print_summary(results: list[dict], total: int) -> None:
    """Print a final summary table to stdout.

    Args:
        results: All run result dicts returned by :func:`_run_one`.
        total: Total number of models attempted.
    """
    completed = [r for r in results if r["status"] == "completed"]
    best = find_best_run(results)

    print(f"\n{'=' * 60}")
    print("SEQUENTIAL TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Completed : {len(completed)}/{total}")
    if best:
        print(f"Best model: {best['run_name']}")
        print(f"Best acc  : {best['best_test_acc'] * 100:.2f}%")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sequential(
    configs: dict[str, ModelConfig],
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
        configs: Mapping of model name → :class:`~src.deep_learning.configs.ModelConfig`.
                 Each :class:`ModelConfig` specifies the required ``sampling``
                 method and optional per-model overrides for ``lr``,
                 ``patience``, ``early_stop_metric``, and ``epochs``.
        n_points: Number of points sampled per mesh (same for all models).
        batch_size: Training batch size (same for all models).
        epochs: Default number of training epochs per model.  Can be
                overridden per-model via :attr:`ModelConfig.epochs`.
        lr: Global learning rate.  ``None`` → auto ``0.001 * (batch_size / 32)``.
            Can be overridden per-model via :attr:`ModelConfig.lr`.
        patience: Global early-stopping patience (epochs without improvement).
                  ``None`` disables early stopping.  Can be overridden
                  per-model via :attr:`ModelConfig.patience`.
        early_stop_metric: Metric watched globally by early stopping —
                           ``"accuracy"``, ``"f1"`` (macro F1), or ``"loss"``.
                           Can be overridden per-model via
                           :attr:`ModelConfig.early_stop_metric`.
        data_dir: Root directory of ModelNet10 data.
        results_dir: Directory where JSON results and plots are saved.
        models_dir: Directory where model checkpoints are saved.

    Returns:
        List of run result dicts (one per model), each with keys:
        ``config``, ``run_name``, ``metrics``, ``timestamp``, ``status``
        (and ``error`` if the run failed).

    Raises:
        KeyError: If any model name in *configs* is not found in
                  :data:`~src.deep_learning.models.ALL_MODELS`.
        TypeError: If any value in *configs* is not a
                   :class:`~src.deep_learning.configs.ModelConfig` instance.
    """
    _validate(configs)

    results_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "sequential").mkdir(parents=True, exist_ok=True)
    Path("runs").mkdir(parents=True, exist_ok=True)

    total = len(configs)
    start_time = datetime.now()

    print(f"\n{'=' * 60}")
    print(f"SEQUENTIAL TRAINER: {total} model(s)")
    print(
        f"  n_points={n_points}  batch_size={batch_size}  epochs={epochs}  "
        f"lr={lr!r}  patience={patience!r}  metric={early_stop_metric!r}"
    )
    print(f"{'=' * 60}\n")

    dataset_cache: dict[tuple[int, str], tuple] = {}
    results: list[dict] = []

    for idx, (model_name, model_cfg) in enumerate(configs.items(), start=1):
        # Resolve per-model overrides (None → fall back to global value)
        effective_lr       = model_cfg.lr                if model_cfg.lr                is not None else lr
        effective_patience = model_cfg.patience          if model_cfg.patience          is not None else patience
        effective_metric   = model_cfg.early_stop_metric if model_cfg.early_stop_metric is not None else early_stop_metric
        effective_epochs   = model_cfg.epochs            if model_cfg.epochs            is not None else epochs

        results.append(_run_one(
            model_name, model_cfg,
            n_points=n_points,
            batch_size=batch_size,
            effective_lr=effective_lr,
            effective_patience=effective_patience,
            effective_metric=effective_metric,
            effective_epochs=effective_epochs,
            dataset_cache=dataset_cache,
            models_dir=models_dir,
            data_dir=data_dir,
            idx=idx,
            total=total,
        ))

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    end_time = datetime.now()
    duration_hours = (end_time - start_time).total_seconds() / 3600

    output = {
        "experiment_metadata": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_hours": round(duration_hours, 2),
            "num_models": total,
            "num_completed": sum(1 for r in results if r["status"] == "completed"),
            "n_points": n_points,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "early_stop_metric": early_stop_metric,
        },
        "runs": results,
        "best_run": find_best_run(results),
    }
    save_json(output, results_dir / "sequential_results.json")

    # ------------------------------------------------------------------
    # Generate tailored plots
    # ------------------------------------------------------------------
    if any(r["status"] == "completed" for r in results):
        plot_sequential_results(results_dir / "sequential_results.json")
    else:
        print("No completed runs — skipping plots.")

    _print_summary(results, total)
    return results
