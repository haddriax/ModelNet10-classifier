"""Sequential training entry point with curated per-model hyperparameters.

Trains each model with its own sampling method, learning rate, patience and
epoch budget rather than running a full Cartesian grid search.  One run per
model, executed sequentially.

Hyperparameters follow each model's original research paper as closely as
possible.  Paper-specific notes are included inline.

To use a custom scheduler or optimizer, pass a factory callable to ModelConfig::

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import StepLR

    ModelConfig(
        sampling="fps",
        optimizer_factory=lambda params, lr: AdamW(params, lr=lr, weight_decay=1e-4),
        scheduler_factory=lambda opt, _: StepLR(opt, step_size=20, gamma=0.7),
    )

The implementation lives in :mod:`src.deep_learning.sequential_trainer`.
"""

import torch
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from src.deep_learning.configs import ModelConfig
from src.deep_learning.sequential_trainer import run_sequential


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Common global training settings (applies to all models unless
    # overridden in a per-model ModelConfig).
    # ------------------------------------------------------------------
    N_POINTS  = 1024
    BATCH_SIZE = 32

    configs: dict[str, ModelConfig] = {
        # --------------------------------------------------------------
        # PointNet — Qi et al., CVPR 2017 (arXiv:1612.00593)
        # Paper config: Adam lr=0.001, exponential LR decay ×0.7 every
        # 200 K steps (≈ 200 epochs at batch 32 / 1024 pts), 250 epochs,
        # uniform sampling, 1024 points.
        # Reference: https://github.com/charlesq34/pointnet train.py
        # --------------------------------------------------------------
        "PointNet": ModelConfig(
            sampling="uniform",
            lr=0.001,
            epochs=250,
            patience=20,
            early_stop_metric="accuracy",
            # Exponential decay: ×0.7 every ~200 k gradient steps.
            # With 1024 pts / batch 32 → ~9 843 steps per epoch → decay
            # every ≈ 20 epochs. gamma = 0.7 ** (1/20) ≈ 0.9827 per epoch.
            scheduler_factory=lambda opt, _: ExponentialLR(opt, gamma=0.9827),
        ),

        # SimplePointNet — minimal PointNet backbone (no transform nets).
        "SimplePointNet": ModelConfig(
            sampling="uniform",
            lr=0.001,
            epochs=250,
            patience=20,
            early_stop_metric="accuracy",
            scheduler_factory=lambda opt, _: ExponentialLR(opt, gamma=0.9827),
        ),

        # --------------------------------------------------------------
        # PointNet++ — Qi et al., NeurIPS 2017 (arXiv:1706.02413)
        # Paper config: Adam lr=0.001, exponential decay ×0.7 every
        # 200 k steps, 250 epochs, batch 32, 1024 points, FPS sampling.
        # The hierarchical SA layers are designed around FPS; using
        # uniform sampling will still work but is sub-optimal.
        # Reference: https://github.com/charlesq34/pointnet2 train.py
        # --------------------------------------------------------------
        "PointNetPP": ModelConfig(
            sampling="fps",
            lr=0.001,
            epochs=250,
            patience=25,
            early_stop_metric="accuracy",
            scheduler_factory=lambda opt, _: ExponentialLR(opt, gamma=0.9827),
        ),

        # --------------------------------------------------------------
        # Point Transformer — Zhao et al., ICCV 2021 (arXiv:2012.09164)
        # Paper config: AdamW, lr=0.001, cosine annealing, weight
        # decay=0.05, 200 epochs, batch 32, 1024 points.
        # AdamW + cosine annealing is standard for transformer models
        # (see also official implementation).
        # Reference: https://arxiv.org/abs/2012.09164
        # --------------------------------------------------------------
        "PointTransformer": ModelConfig(
            sampling="fps",
            lr=0.001,
            epochs=200,
            patience=30,
            early_stop_metric="accuracy",
            optimizer_factory=lambda params, lr: torch.optim.AdamW(
                params, lr=lr, weight_decay=0.05
            ),
            scheduler_factory=lambda opt, epochs: torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=1e-6
            ),
        ),

        # --------------------------------------------------------------
        # DGCNN — Wang et al., TOG 2019 (arXiv:1801.07829)
        # Paper config: Adam lr=0.001, step decay ×0.5 every 20 epochs,
        # 200 epochs, batch 32, 1024 uniform-sampled points.
        # Reference: https://github.com/WangYueFt/dgcnn train.py
        # --------------------------------------------------------------
        "DGCNN": ModelConfig(
            sampling="uniform",
            lr=0.001,
            epochs=200,
            patience=30,
            early_stop_metric="accuracy",
            scheduler_factory=lambda opt, _: StepLR(opt, step_size=20, gamma=0.5),
        ),
    }

    run_sequential(
        configs,
        n_points=N_POINTS,
        batch_size=BATCH_SIZE,
        epochs=250,   # global fallback (each model overrides via ModelConfig.epochs)
        early_stop_metric="accuracy",
    )
