"""Grid search ablation study for point cloud classification.

Provides GridSearchConfig, AblationConfig, and GridSearch to orchestrate
multi-configuration training runs with crash recovery and results persistence.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Type

from torch import nn
from torch.utils.data import Dataset

from src.config import MODELS_DIR, RESULTS_DIR
from src.deep_learning.training import ModelTrainer
from src.geometry import Sampling


# Type alias: factory that creates (train_dataset, test_dataset) for given params
DatasetFactory = Callable[[int, Sampling], tuple[Dataset, Dataset]]


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


@dataclass
class GridSearchConfig:
    """Configuration for a full grid search experiment.

    Defines the parameter grid. Use generate_configs() to produce
    the Cartesian product of all combinations.

    Args:
        model_classes: List of model classes to evaluate
        sampling_methods: List of point sampling strategies
        n_points_list: List of point cloud resolutions
        batch_sizes: List of batch sizes
        epochs: Number of training epochs per run
    """
    model_classes: list[Type[nn.Module]]
    sampling_methods: list[Sampling]
    n_points_list: list[int]
    batch_sizes: list[int]
    epochs: int = 50

    def generate_configs(self) -> list[AblationConfig]:
        """Generate Cartesian product of all grid parameters."""
        return [
            AblationConfig(model, sampling, n_pts, bs, self.epochs)
            for model, sampling, n_pts, bs
            in product(self.model_classes, self.sampling_methods,
                       self.n_points_list, self.batch_sizes)
        ]

    def to_dict(self) -> dict:
        """JSON-serializable representation for experiment metadata."""
        return {
            "models": [m.__name__ for m in self.model_classes],
            "sampling_methods": [s.value for s in self.sampling_methods],
            "n_points": self.n_points_list,
            "batch_sizes": self.batch_sizes,
            "epochs": self.epochs,
        }


# ============================================================================
# Grid Search
# ============================================================================

class GridSearch:
    """Runs a grid search ablation study over model configurations.

    Accepts a GridSearchConfig and a dataset factory, then orchestrates
    multiple ModelTrainer runs with crash recovery via intermediate JSON saves.

    Args:
        grid_config: GridSearchConfig defining the parameter grid
        dataset_factory: Callable (n_points, Sampling) -> (train_ds, test_ds)
        results_dir: Directory for JSON results
        models_dir: Directory for model checkpoints
    """

    def __init__(
        self,
        grid_config: GridSearchConfig,
        dataset_factory: DatasetFactory,
        results_dir: Path = RESULTS_DIR,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        self.grid_config = grid_config
        self._dataset_factory = dataset_factory
        self._dataset_cache: dict[tuple[int, str], tuple[Dataset, Dataset]] = {}
        self.results_dir = results_dir
        self.models_dir = models_dir

        self._configs = grid_config.generate_configs()
        self._results: list[dict] = []
        self._start_time: datetime | None = None

    @property
    def num_configs(self) -> int:
        """Total number of configurations in the grid."""
        return len(self._configs)

    def run(self) -> list[dict]:
        """Execute all grid search configurations sequentially.

        Returns:
            List of run result dicts, each with keys:
            config, run_name, metrics, timestamp, status (and error if failed)
        """
        self._start_time = datetime.now()
        self._setup_directories()

        print(f"\n{'=' * 60}")
        print(f"ABLATION STUDY: {self.num_configs} configurations")
        print(f"{'=' * 60}\n")

        for i, config in enumerate(self._configs, 1):
            run_result = self._run_single(config, index=i)
            self._results.append(run_result)
            self._save_intermediate()

        return self._results

    def save_results(self) -> Path:
        """Save complete ablation results to JSON file.

        Returns:
            Path to saved JSON file
        """
        end_time = datetime.now()
        duration_hours = (end_time - self._start_time).total_seconds() / 3600

        completed_runs = [r for r in self._results if r.get("status") == "completed"]
        best_run = None
        if completed_runs:
            best = max(completed_runs, key=lambda r: r["metrics"]["best_test_acc"])
            best_run = {
                "run_name": best["run_name"],
                "best_test_acc": best["metrics"]["best_test_acc"],
            }

        output = {
            "experiment_metadata": {
                "start_time": self._start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_hours": round(duration_hours, 2),
                "num_configurations": len(self._results),
                "num_completed": len(completed_runs),
                "grid_parameters": self.grid_config.to_dict(),
            },
            "runs": self._results,
            "best_run": best_run,
        }

        output_path = self.results_dir / "ablation_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path

    # -- Private methods --

    def _get_datasets(self, config: AblationConfig) -> tuple[Dataset, Dataset]:
        """Get or create datasets, cached by (n_points, sampling_method)."""
        key = (config.n_points, config.sampling_method.value)
        if key not in self._dataset_cache:
            self._dataset_cache[key] = self._dataset_factory(
                config.n_points, config.sampling_method
            )
        return self._dataset_cache[key]

    def _run_single(self, config: AblationConfig, index: int) -> dict:
        """Train one configuration and return result dict."""
        run_name = config.get_run_name()
        save_path = self.models_dir / "ablation" / f"{run_name}.pth"

        print(f"\n[{index}/{self.num_configs}] Running: {run_name}")
        print(f"  Model:    {config.model_class.__name__}")
        print(f"  Sampling: {config.sampling_method.value}")
        print(f"  Points:   {config.n_points}")
        print(f"  Batch:    {config.batch_size}")
        print(f"  Epochs:   {config.epochs}")

        timestamp = datetime.now().isoformat()

        try:
            train_ds, test_ds = self._get_datasets(config)
            num_classes = len(train_ds.class_to_idx)
            model = config.model_class(num_classes=num_classes)

            trainer = ModelTrainer(
                train_ds,
                test_ds,
                model=model,
                batch_size=config.batch_size,
                save_model=save_path,
                experiment_name=f"ablation_{run_name}",
            )

            metrics = trainer.train(epochs=config.epochs, resume=False)

            print(f"  \u2713 Best acc: {metrics['best_test_acc']:.4f}  "
                  f"F1: {metrics['macro_f1']:.4f}")

            return {
                "config": config.to_dict(),
                "run_name": run_name,
                "metrics": dict(metrics),
                "timestamp": timestamp,
                "status": "completed",
            }

        except Exception as e:
            print(f"  \u2717 ERROR: {e}")
            return {
                "config": config.to_dict(),
                "run_name": run_name,
                "metrics": None,
                "timestamp": timestamp,
                "status": "failed",
                "error": str(e),
            }

    def _save_intermediate(self) -> None:
        """Save intermediate results to disk for crash recovery."""
        path = self.results_dir / "ablation_results_intermediate.json"
        with open(path, 'w') as f:
            json.dump({"runs": self._results}, f, indent=2)

    def _setup_directories(self) -> None:
        """Create output directories if they don't exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "ablation").mkdir(parents=True, exist_ok=True)
