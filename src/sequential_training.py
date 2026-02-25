"""Sequential training entry point with curated per-model hyperparameters.

Trains each model with its own sampling method, learning rate, patience and
epoch budget rather than running a full Cartesian grid search.  One run per
model, executed sequentially.
"""

from src.deep_learning.configs import ModelConfig
from src.deep_learning.sequential_trainer import run_sequential


if __name__ == "__main__":
    configs: dict[str, ModelConfig] = {
        "PointNet":         ModelConfig(
            sampling="uniform",
            lr=0.001,
            patience=10,
            early_stop_metric="loss",
            epochs=100
        ),
        # "SimplePointNet": ModelConfig(sampling="uniform"),
        # "DGCNN":          ModelConfig(sampling="fps"),
        "PointNetPP":       ModelConfig(
            sampling="uniform",
            lr=0.001,
            patience=12,
            early_stop_metric="loss",
            epochs=200,
        ),
        # "PointTransformer": ModelConfig(sampling="uniform"),
    }

    run_sequential(
        configs,
        n_points=2048,
        batch_size=32,
        epochs=50,
    )
