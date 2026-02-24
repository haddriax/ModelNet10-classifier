"""Training entry point with grid search ablation study.

Supports running a full ablation study over
(Model Architecture x Sampling Method x n_points x batch_size).
Results are saved as JSON and matplotlib plots.
"""

from src.config import MODELS_DIR, RESULTS_DIR
from src.deep_learning.dataset_factory import make_datasets
from src.deep_learning.grid_search import GridSearch, GridSearchConfig
from src.deep_learning.models import ALL_MODELS
from src.deep_learning.plotting import create_ablation_plots
from src.geometry import Sampling


if __name__ == "__main__":
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
        dataset_factory=make_datasets,
        results_dir=RESULTS_DIR,
        models_dir=MODELS_DIR,
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
