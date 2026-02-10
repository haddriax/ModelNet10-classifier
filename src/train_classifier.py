"""Training entry point with grid search ablation study.

Supports training a single model or running a full ablation study over
(Model Architecture × Sampling Method × n_points × batch_size).
Results are saved as JSON and matplotlib plots.
"""

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Type
import json

from torch import nn

from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.dataset.base_modelnet_dataset import PointCloudDataset
from src.deep_learning.models import ALL_MODELS
from src.deep_learning.training import ModelTrainer, TrainingResults
from src.geometry.Mesh_3D import Sampling


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation run."""
    model_class: Type[nn.Module]
    sampling_method: Sampling
    n_points: int
    batch_size: int
    epochs: int = 50

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "model": self.model_class.__name__,
            "sampling_method": self.sampling_method.value,
            "n_points": self.n_points,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    def get_run_name(self) -> str:
        """Generate unique run identifier."""
        return (f"{self.model_class.__name__}"
                f"_{self.sampling_method.value}"
                f"_pts{self.n_points}"
                f"_bs{self.batch_size}")


# ============================================================================
# Training
# ============================================================================

def train_model(
    model_class: Type[nn.Module],
    epochs: int = 50,
    resume: bool = False,
    batch_size: int = 32,
    n_points: int = 1024,
    sampling_method: Sampling = Sampling.UNIFORM,
    data_dir: Path | None = None,
    save_path: Path | None = None,
    experiment_name: str | None = None,
) -> TrainingResults:
    """Train a point cloud classification model.

    Args:
        model_class: PyTorch model class to instantiate
        epochs: Number of training epochs
        resume: Whether to resume from checkpoint
        batch_size: Training batch size
        n_points: Number of points to sample from each mesh
        sampling_method: Point sampling strategy
        data_dir: Path to ModelNet data directory
        save_path: Path to save model checkpoint
        experiment_name: Name for TensorBoard logs

    Returns:
        TrainingResults dict with all metrics
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if save_path is None:
        save_path = MODELS_DIR / f"{model_class.__name__}_modelnet10.pth"
    if experiment_name is None:
        experiment_name = (f"{model_class.__name__}"
                           f"_{sampling_method.value}"
                           f"_pts{n_points}"
                           f"_bs{batch_size}")

    print(f"Using data directory: {data_dir.resolve()}")

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

    num_classes = len(train_ds.class_to_idx)
    model = model_class(num_classes=num_classes)

    trainer = ModelTrainer(
        train_ds,
        test_ds,
        model=model,
        batch_size=batch_size,
        save_model=save_path,
        experiment_name=experiment_name,
    )

    return trainer.train(epochs=epochs, resume=resume)


# ============================================================================
# Grid Search
# ============================================================================

def generate_grid_configs(
    model_classes: list[Type[nn.Module]],
    sampling_methods: list[Sampling],
    n_points_list: list[int],
    batch_sizes: list[int],
    epochs: int = 50,
) -> list[AblationConfig]:
    """Generate all configurations for grid search."""
    return [
        AblationConfig(model, sampling, n_pts, bs, epochs)
        for model, sampling, n_pts, bs
        in product(model_classes, sampling_methods, n_points_list, batch_sizes)
    ]


def run_ablation_study(
    configs: list[AblationConfig],
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> list[dict]:
    """Run grid search ablation study over all configurations.

    Args:
        configs: List of ablation configurations
        data_dir: Path to data directory
        results_dir: Path to save intermediate results

    Returns:
        List of run results (each with config, run_name, metrics, timestamp)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if results_dir is None:
        results_dir = RESULTS_DIR

    results_dir.mkdir(parents=True, exist_ok=True)
    ablation_models_dir = MODELS_DIR / "ablation"
    ablation_models_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    print(f"\n{'=' * 60}")
    print(f"ABLATION STUDY: {len(configs)} configurations")
    print(f"{'=' * 60}\n")

    for i, config in enumerate(configs, 1):
        run_name = config.get_run_name()
        save_path = ablation_models_dir / f"{run_name}.pth"

        print(f"\n[{i}/{len(configs)}] Running: {run_name}")
        print(f"  Model:    {config.model_class.__name__}")
        print(f"  Sampling: {config.sampling_method.value}")
        print(f"  Points:   {config.n_points}")
        print(f"  Batch:    {config.batch_size}")
        print(f"  Epochs:   {config.epochs}")

        timestamp = datetime.now().isoformat()

        try:
            metrics = train_model(
                model_class=config.model_class,
                epochs=config.epochs,
                batch_size=config.batch_size,
                n_points=config.n_points,
                sampling_method=config.sampling_method,
                data_dir=data_dir,
                save_path=save_path,
                experiment_name=f"ablation_{run_name}",
            )

            run_result = {
                "config": config.to_dict(),
                "run_name": run_name,
                "metrics": dict(metrics),
                "timestamp": timestamp,
                "status": "completed",
            }

            print(f"  ✓ Best test accuracy: {metrics['best_test_acc']:.4f}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            run_result = {
                "config": config.to_dict(),
                "run_name": run_name,
                "metrics": None,
                "timestamp": timestamp,
                "status": "failed",
                "error": str(e),
            }

        all_results.append(run_result)

        # Intermediate save for crash recovery
        _save_intermediate_results(all_results, results_dir)

    return all_results


def _save_intermediate_results(results: list[dict], results_dir: Path) -> None:
    """Save intermediate results to disk (for crash recovery)."""
    path = results_dir / "ablation_results_intermediate.json"
    with open(path, 'w') as f:
        json.dump({"runs": results}, f, indent=2)


# ============================================================================
# Results Saving
# ============================================================================

def save_ablation_results(
    results: list[dict],
    grid_params: dict,
    start_time: datetime,
    results_dir: Path,
) -> Path:
    """Save complete ablation results to JSON file.

    Returns:
        Path to saved JSON file
    """
    end_time = datetime.now()
    duration_hours = (end_time - start_time).total_seconds() / 3600

    completed_runs = [r for r in results if r.get("status") == "completed"]
    best_run = None
    if completed_runs:
        best = max(completed_runs, key=lambda r: r["metrics"]["best_test_acc"])
        best_run = {
            "run_name": best["run_name"],
            "best_test_acc": best["metrics"]["best_test_acc"],
        }

    output = {
        "experiment_metadata": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_hours": round(duration_hours, 2),
            "num_configurations": len(results),
            "num_completed": len(completed_runs),
            "grid_parameters": grid_params,
        },
        "runs": results,
        "best_run": best_run,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


# ============================================================================
# Plotting
# ============================================================================

def create_ablation_plots(results_path: Path, output_dir: Path | None = None) -> None:
    """Generate all ablation study plots from results JSON.

    Args:
        results_path: Path to ablation_results.json
        output_dir: Directory to save plots (default: same as results)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = results_path.parent

    with open(results_path) as f:
        data = json.load(f)

    runs = [r for r in data["runs"] if r.get("status") == "completed"]

    if not runs:
        print("No completed runs to plot")
        return

    plot_accuracy_comparison(runs, output_dir, plt)
    plot_npoints_effect(runs, output_dir, plt)
    plot_batchsize_effect(runs, output_dir, plt)
    plot_sampling_comparison(runs, output_dir, plt)
    plot_model_heatmap(runs, output_dir, plt)

    print(f"Plots saved to: {output_dir}")


def _get_model_colors() -> dict[str, str]:
    """Consistent color palette for model architectures."""
    return {
        "SimplePointNet": "#3498db",
        "DGCNN": "#e74c3c",
        "PointNetPP": "#2ecc71",
        "PointTransformer": "#9b59b6",
    }


def plot_accuracy_comparison(runs: list[dict], output_dir: Path, plt) -> None:
    """Bar chart comparing best test accuracy across all configurations."""
    import numpy as np

    runs_sorted = sorted(runs, key=lambda r: r["metrics"]["best_test_acc"], reverse=True)
    model_colors = _get_model_colors()

    run_names = [r["run_name"] for r in runs_sorted]
    accuracies = [r["metrics"]["best_test_acc"] * 100 for r in runs_sorted]
    colors = [model_colors.get(r["config"]["model"], "#999999") for r in runs_sorted]

    fig, ax = plt.subplots(figsize=(14, max(6, len(runs_sorted) * 0.35)))

    y_pos = np.arange(len(run_names))
    bars = ax.barh(y_pos, accuracies, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(run_names, fontsize=7)
    ax.set_xlabel('Best Test Accuracy (%)')
    ax.set_title('Ablation Study: Test Accuracy Comparison')
    ax.invert_yaxis()

    # Value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1f}%', ha='left', va='center', fontsize=7)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in model_colors.values()]
    ax.legend(handles, model_colors.keys(), loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()


def plot_npoints_effect(runs: list[dict], output_dir: Path, plt) -> None:
    """Line plot: effect of n_points on accuracy, one line per model."""
    model_colors = _get_model_colors()
    markers = {"SimplePointNet": "o", "DGCNN": "s", "PointNetPP": "^", "PointTransformer": "D"}
    models = sorted(set(r["config"]["model"] for r in runs))

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in models:
        model_runs = [r for r in runs if r["config"]["model"] == model_name]

        # Group by n_points, average across sampling methods & batch sizes
        npoints_acc: dict[int, list[float]] = {}
        for r in model_runs:
            n = r["config"]["n_points"]
            acc = r["metrics"]["best_test_acc"] * 100
            npoints_acc.setdefault(n, []).append(acc)

        n_points = sorted(npoints_acc.keys())
        mean_acc = [sum(npoints_acc[n]) / len(npoints_acc[n]) for n in n_points]

        ax.plot(n_points, mean_acc,
                marker=markers.get(model_name, "o"),
                color=model_colors.get(model_name, "#999999"),
                label=model_name,
                linewidth=2, markersize=8)

    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Effect of Point Cloud Resolution on Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'npoints_effect.png', dpi=150)
    plt.close()


def plot_batchsize_effect(runs: list[dict], output_dir: Path, plt) -> None:
    """Grouped bar chart: effect of batch size, grouped by model."""
    import numpy as np

    model_colors = _get_model_colors()
    models = sorted(set(r["config"]["model"] for r in runs))
    batch_sizes = sorted(set(r["config"]["batch_size"] for r in runs))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(batch_sizes))
    n_models = len(models)
    width = 0.8 / n_models

    for i, model_name in enumerate(models):
        model_runs = [r for r in runs if r["config"]["model"] == model_name]

        bs_acc: dict[int, list[float]] = {}
        for r in model_runs:
            bs = r["config"]["batch_size"]
            acc = r["metrics"]["best_test_acc"] * 100
            bs_acc.setdefault(bs, []).append(acc)

        mean_acc = [sum(bs_acc.get(bs, [0])) / max(len(bs_acc.get(bs, [1])), 1)
                    for bs in batch_sizes]

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, mean_acc, width,
               label=model_name,
               color=model_colors.get(model_name, "#999999"))

    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Effect of Batch Size on Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'batchsize_effect.png', dpi=150)
    plt.close()


def plot_sampling_comparison(runs: list[dict], output_dir: Path, plt) -> None:
    """Grouped bar chart: sampling method comparison per model."""
    import numpy as np

    model_colors = {"uniform": "#3498db", "fps": "#e74c3c"}
    models = sorted(set(r["config"]["model"] for r in runs))
    sampling_methods = sorted(set(r["config"]["sampling_method"] for r in runs))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    n_methods = len(sampling_methods)
    width = 0.8 / n_methods

    for i, method in enumerate(sampling_methods):
        method_runs = [r for r in runs if r["config"]["sampling_method"] == method]

        model_acc: dict[str, list[float]] = {}
        for r in method_runs:
            m = r["config"]["model"]
            acc = r["metrics"]["best_test_acc"] * 100
            model_acc.setdefault(m, []).append(acc)

        mean_acc = [sum(model_acc.get(m, [0])) / max(len(model_acc.get(m, [1])), 1)
                    for m in models]

        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, mean_acc, width,
                       label=method.upper(),
                       color=model_colors.get(method, "#999999"))

        # Value labels
        for bar, acc in zip(bars, mean_acc):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{acc:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Sampling Method Comparison per Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'sampling_comparison.png', dpi=150)
    plt.close()


def plot_model_heatmap(runs: list[dict], output_dir: Path, plt) -> None:
    """Heatmap: models × (sampling × n_points), averaged across batch sizes."""
    import numpy as np

    models = sorted(set(r["config"]["model"] for r in runs))
    sampling_methods = sorted(set(r["config"]["sampling_method"] for r in runs))
    n_points_list = sorted(set(r["config"]["n_points"] for r in runs))

    # Column labels: "uniform_256", "uniform_512", etc.
    col_labels = [f"{s}_{n}" for s in sampling_methods for n in n_points_list]
    n_cols = len(col_labels)
    n_rows = len(models)

    # Build accuracy matrix
    matrix = np.zeros((n_rows, n_cols))
    for i, model_name in enumerate(models):
        for j, (method, n_pts) in enumerate(product(sampling_methods, n_points_list)):
            matching = [r for r in runs
                        if r["config"]["model"] == model_name
                        and r["config"]["sampling_method"] == method
                        and r["config"]["n_points"] == n_pts]
            if matching:
                matrix[i, j] = sum(r["metrics"]["best_test_acc"] for r in matching) / len(matching) * 100

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.2), max(4, n_rows * 0.8)))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Best Test Accuracy (%)')

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(models, fontsize=9)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f'{matrix[i, j]:.1f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, j] > (matrix.max() + matrix.min()) / 2 else 'black')

    ax.set_title('Model × Configuration Interaction (avg over batch sizes)')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_heatmap.png', dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Grid search parameters
    GRID = {
        "models": list(ALL_MODELS.values()),
        "sampling_methods": [Sampling.UNIFORM, Sampling.FARTHEST_POINT],
        "n_points": [256, 512, 1024],
        "batch_sizes": [32, 128],
        "epochs": 50,
    }

    configs = generate_grid_configs(
        model_classes=GRID["models"],
        sampling_methods=GRID["sampling_methods"],
        n_points_list=GRID["n_points"],
        batch_sizes=GRID["batch_sizes"],
        epochs=GRID["epochs"],
    )

    print(f"Generated {len(configs)} configurations for ablation study")

    # Run ablation
    start_time = datetime.now()
    results = run_ablation_study(configs=configs)

    # Save results
    grid_params_serializable = {
        "models": [m.__name__ for m in GRID["models"]],
        "sampling_methods": [s.value for s in GRID["sampling_methods"]],
        "n_points": GRID["n_points"],
        "batch_sizes": GRID["batch_sizes"],
        "epochs": GRID["epochs"],
    }

    results_path = save_ablation_results(
        results=results,
        grid_params=grid_params_serializable,
        start_time=start_time,
        results_dir=RESULTS_DIR,
    )

    # Generate plots
    create_ablation_plots(results_path)

    # Print summary
    completed = [r for r in results if r.get("status") == "completed"]
    if completed:
        best = max(completed, key=lambda r: r["metrics"]["best_test_acc"])
        print(f"\n{'=' * 60}")
        print(f"ABLATION STUDY COMPLETE")
        print(f"{'=' * 60}")
        print(f"Completed: {len(completed)}/{len(results)} runs")
        print(f"Best configuration: {best['run_name']}")
        print(f"Best test accuracy:  {best['metrics']['best_test_acc'] * 100:.2f}%")
        print(f"{'=' * 60}")
