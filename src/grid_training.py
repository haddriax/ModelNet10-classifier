"""Training entry point with grid search ablation study.

Supports running a full ablation study over
(Model Architecture x Sampling Method x n_points x batch_size)
on either ModelNet10 or ModelNet40.

Results are saved to a timestamped directory so past runs are never
overwritten.  JSON + matplotlib plots are produced automatically.

Usage::

    # ModelNet10 (default)
    python -m src.grid_training --dataset modelnet10

    # ModelNet40
    python -m src.grid_training --dataset modelnet40
"""

from functools import partial

from src.deep_learning.dataset_factory import make_datasets
from src.deep_learning.grid_search import GridSearch, GridSearchConfig
from src.deep_learning.models import ALL_MODELS
from src.deep_learning.plotting import create_ablation_plots
from src.geometry import Sampling


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    from src.config import DATA_DIR, MODELNET40_DIR, MODELS_DIR, RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Ablation grid search on ModelNet10 or ModelNet40."
    )
    parser.add_argument(
        "--dataset",
        choices=["modelnet10", "modelnet40"],
        default="modelnet10",
        help="Dataset to train on (default: modelnet10).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if args.dataset == "modelnet40":
        data_dir    = MODELNET40_DIR
        results_dir = RESULTS_DIR / "grid" / "modelnet40" / timestamp
        models_dir  = MODELS_DIR  / "grid" / "modelnet40" / timestamp
    else:
        data_dir    = DATA_DIR
        results_dir = RESULTS_DIR / "grid" / "modelnet10" / timestamp
        models_dir  = MODELS_DIR  / "grid" / "modelnet10" / timestamp

    # Pre-bind data_dir so the factory matches GridSearch's (n_points, Sampling) signature.
    dataset_factory = partial(make_datasets, data_dir=data_dir)

    config = GridSearchConfig(
        model_classes=[
            ALL_MODELS['PointNet'],
            ALL_MODELS['PointNetPP'],
        ],
        sampling_methods=[Sampling.UNIFORM, Sampling.FARTHEST_POINT],
        n_points_list=[2048],
        batch_sizes=[32],
        epochs=50,
    )

    search = GridSearch(
        grid_config=config,
        dataset_factory=dataset_factory,
        results_dir=results_dir,
        models_dir=models_dir,
    )

    print(f"Generated {search.num_configs} configurations for ablation study")

    results = search.run()
    results_path = search.save_results()
    create_ablation_plots(results_path)

    completed = [r for r in results if r.get("status") == "completed"]
    if completed:
        best = max(completed, key=lambda r: r["metrics"]["best_test_acc"])
        print(f"\n{'=' * 60}")
        print(f"ABLATION STUDY COMPLETE")
        print(f"{'=' * 60}")
        print(f"Completed: {len(completed)}/{len(results)} runs")
        print(f"Best configuration: {best['run_name']}")
        print(f"Best test accuracy:  {best['metrics']['best_test_acc'] * 100:.2f}%")
        print(f"Macro F1:            {best['metrics']['macro_f1'] * 100:.2f}%")
        print(f"Macro Precision:     {best['metrics']['macro_precision'] * 100:.2f}%")
        print(f"Macro Recall:        {best['metrics']['macro_recall'] * 100:.2f}%")
        print(f"{'=' * 60}")
