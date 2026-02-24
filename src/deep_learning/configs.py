"""Configuration dataclasses for sequential training.

Provides :class:`ModelConfig` — a typed, self-validating per-model
configuration used by :func:`src.deep_learning.sequential_trainer.run_sequential`.

Example::

    from src.deep_learning.configs import ModelConfig

    configs = {
        "PointNet":         ModelConfig(sampling="uniform"),
        "DGCNN":            ModelConfig(sampling="fps", lr=5e-4),
        "PointNetPP":       ModelConfig(sampling="fps", patience=10, epochs=100),
        "PointTransformer": ModelConfig(sampling="fps", early_stop_metric="f1"),
    }
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_SAMPLING: frozenset[str] = frozenset({"uniform", "fps", "poisson"})
_VALID_ES_METRICS: frozenset[str] = frozenset({"accuracy", "f1", "loss"})


@dataclass
class ModelConfig:
    """Per-model configuration for the sequential trainer.

    All optional fields default to ``None``, which means "use the global
    value passed to :func:`~src.deep_learning.sequential_trainer.run_sequential`".

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

    Raises:
        ValueError: On construction if any argument value is invalid.

    Examples:
        Minimal config (sampling is the only required field)::

            cfg = ModelConfig(sampling="fps")

        Full per-model override::

            cfg = ModelConfig(
                sampling="fps",
                lr=5e-4,
                patience=10,
                early_stop_metric="f1",
                epochs=100,
            )
    """

    sampling: str
    lr: float | None = None
    patience: int | None = None
    early_stop_metric: str | None = None  # None → use run_sequential() global
    epochs: int | None = None             # None → use run_sequential() global

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
