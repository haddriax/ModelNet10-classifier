"""Training entry point with grid search ablation study.

Supports running a full ablation study over
(Model Architecture x Sampling Method x n_points x batch_size).
Results are saved as JSON and matplotlib plots.
"""

from pathlib import Path

from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.dataset.base_modelnet_dataset import PointCloudDataset
from src.deep_learning.grid_search import GridSearch, GridSearchConfig
from src.deep_learning.models import ALL_MODELS
from src.deep_learning.plotting import create_ablation_plots
from src.geometry.Mesh_3D import Sampling


def make_datasets(
    n_points: int,
    sampling_method: Sampling,
    data_dir: Path = DATA_DIR,
) -> tuple[PointCloudDataset, PointCloudDataset]:
    """Create train/test PointCloudDataset pair.

    Args:
        n_points: Number of points to sample from each mesh
        sampling_method: Point sampling strategy
        data_dir: Path to ModelNet data directory

    Returns:
        (train_dataset, test_dataset) tuple
    """
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


if __name__ == "__main__":
    # Grid search parameters
    config = GridSearchConfig(
        model_classes=list(ALL_MODELS.values()),
        sampling_methods=[Sampling.UNIFORM],
        n_points_list=[512],
        batch_sizes=[32],
        epochs=10,
    )

    search = GridSearch(
        grid_config=config,
        dataset_factory=make_datasets,
        results_dir=RESULTS_DIR,
        models_dir=MODELS_DIR,
    )

    print(f"Generated {search.num_configs} configurations for ablation study")

    # Run ablation
    results = search.run()

    # Save results
    results_path = search.save_results()

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
        print(f"Macro F1:            {best['metrics']['macro_f1'] * 100:.2f}%")
        print(f"Macro Precision:     {best['metrics']['macro_precision'] * 100:.2f}%")
        print(f"Macro Recall:        {best['metrics']['macro_recall'] * 100:.2f}%")
        print(f"{'=' * 60}")
