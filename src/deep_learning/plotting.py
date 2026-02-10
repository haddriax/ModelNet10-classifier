"""Ablation study plotting utilities.

Generates comparison plots from grid search results JSON.
All functions read from the standardized results format and produce PNG files.
"""

import json
from itertools import product
from pathlib import Path


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
    """Heatmap: models x (sampling x n_points), averaged across batch sizes."""
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

    ax.set_title('Model x Configuration Interaction (avg over batch sizes)')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_heatmap.png', dpi=150)
    plt.close()
