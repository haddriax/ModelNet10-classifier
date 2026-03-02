"""Configuration dataclasses for sequential training.

Provides :class:`ModelConfig` — a typed, self-validating per-model
configuration used by :func:`src.deep_learning.sequential_trainer.run_sequential`.

Example::

    from src.deep_learning.configs import ModelConfig
    from torch.optim.lr_scheduler import StepLR

    configs = {
        "PointNet":         ModelConfig(sampling="uniform"),
        "DGCNN":            ModelConfig(sampling="fps", lr=5e-4),
        "PointNetPP":       ModelConfig(sampling="fps", patience=10, epochs=100),
        "PointTransformer": ModelConfig(
            sampling="fps",
            early_stop_metric="f1",
            scheduler_factory=lambda opt, _: StepLR(opt, step_size=20, gamma=0.7),
        ),
    }
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

_VALID_SAMPLING: frozenset[str] = frozenset({"uniform", "fps", "poisson"})
_VALID_ES_METRICS: frozenset[str] = frozenset({"accuracy", "f1", "loss"})

# ---------------------------------------------------------------------------
# Factory type aliases
# ---------------------------------------------------------------------------

OptimizerFactory = Callable[..., "Optimizer"]
"""Callable ``(parameters, lr: float) -> Optimizer``.

Example::

    import torch
    opt_factory = lambda params, lr: torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
"""

SchedulerFactory = Callable[..., "LRScheduler"]
"""Callable ``(optimizer, epochs_remaining: int) -> LRScheduler``.

The second argument (``epochs_remaining``) is provided so that schedulers
like :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` can use the
correct ``T_max``.  Custom factories that don't need it can use ``_``::

    from torch.optim.lr_scheduler import StepLR
    sched_factory = lambda opt, _: StepLR(opt, step_size=20, gamma=0.7)
"""


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Per-model configuration for the sequential trainer.

    All optional fields default to ``None``, which means "use the global
    value passed to :func:`~src.deep_learning.sequential_trainer.run_sequential`"
    (or the :class:`~src.deep_learning.model_trainer.ModelTrainer` default
    for factory fields).

    Args:
        sampling: Point-cloud sampling method.  Must be one of
                  ``"uniform"``, ``"fps"``, or ``"poisson"``.  **Required.**
        lr: Learning-rate override.  ``None`` → auto-formula
            ``0.001 * (batch_size / 32)``.
        patience: Early-stopping patience in epochs (number of consecutive
                  epochs without improvement before stopping).
                  ``None`` disables early stopping for this model.
        early_stop_metric: Metric monitored by early stopping —
                           ``"accuracy"`` (higher is better),
                           ``"f1"`` (macro F1, higher is better), or
                           ``"loss"`` (lower is better).
                           ``None`` → use the global value from
                           :func:`~src.deep_learning.sequential_trainer.run_sequential`.
        epochs: Maximum number of training epochs for this model.
                ``None`` → use the global value from
                :func:`~src.deep_learning.sequential_trainer.run_sequential`.
        optimizer_factory: Callable ``(parameters, lr) -> Optimizer``.
                           ``None`` → :class:`torch.optim.Adam` with the
                           resolved learning rate.  See :data:`OptimizerFactory`.
        scheduler_factory: Callable ``(optimizer, epochs_remaining) -> LRScheduler``.
                           ``None`` → :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`
                           with ``T_max=epochs_remaining`` and ``eta_min=1e-5``.
                           See :data:`SchedulerFactory`.

    Raises:
        ValueError: On construction if any argument value is invalid.

    Examples:
        Minimal config (sampling is the only required field)::

            cfg = ModelConfig(sampling="fps")

        Full per-model override with a custom scheduler::

            from torch.optim.lr_scheduler import StepLR

            cfg = ModelConfig(
                sampling="fps",
                lr=5e-4,
                patience=10,
                early_stop_metric="f1",
                epochs=100,
                scheduler_factory=lambda opt, _: StepLR(opt, step_size=20, gamma=0.7),
            )
    """

    sampling: str
    lr: float | None = None
    patience: int | None = None
    early_stop_metric: str | None = None  # None → use run_sequential() global
    epochs: int | None = None             # None → use run_sequential() global
    optimizer_factory: OptimizerFactory | None = field(default=None, repr=False)
    scheduler_factory: SchedulerFactory | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.sampling not in _VALID_SAMPLING:
            raise ValueError(
                f"sampling must be one of {sorted(_VALID_SAMPLING)}, "
                f"got {self.sampling!r}"
            )
        if self.lr is not None and not (
            isinstance(self.lr, (int, float)) and self.lr > 0
        ):
            raise ValueError(
                f"lr must be a positive number, got {self.lr!r}"
            )
        if self.patience is not None and not (
            isinstance(self.patience, int) and self.patience >= 1
        ):
            raise ValueError(
                f"patience must be a positive integer, got {self.patience!r}"
            )
        if (
            self.early_stop_metric is not None
            and self.early_stop_metric not in _VALID_ES_METRICS
        ):
            raise ValueError(
                f"early_stop_metric must be one of {sorted(_VALID_ES_METRICS)}, "
                f"got {self.early_stop_metric!r}"
            )
        if self.epochs is not None and not (
            isinstance(self.epochs, int) and self.epochs >= 1
        ):
            raise ValueError(
                f"epochs must be a positive integer, got {self.epochs!r}"
            )
        if self.optimizer_factory is not None and not callable(self.optimizer_factory):
            raise ValueError(
                f"optimizer_factory must be callable, "
                f"got {type(self.optimizer_factory).__name__!r}"
            )
        if self.scheduler_factory is not None and not callable(self.scheduler_factory):
            raise ValueError(
                f"scheduler_factory must be callable, "
                f"got {type(self.scheduler_factory).__name__!r}"
            )
