"""Shared result-handling utilities for training pipelines.

Used by both :mod:`~src.deep_learning.sequential_trainer` and
:mod:`~src.deep_learning.grid_search` to avoid duplicating JSON
serialisation and best-run discovery logic.
"""

import json
from pathlib import Path


def find_best_run(results: list[dict]) -> dict | None:
    """Return a summary dict for the run with the highest ``best_test_acc``.

    Args:
        results: List of run result dicts.  Each dict must have a
            ``"status"`` key and, when ``status == "completed"``, a
            ``"metrics"`` sub-dict containing ``"best_test_acc"``.

    Returns:
        ``{"run_name": str, "best_test_acc": float}`` for the best
        completed run, or ``None`` if there are no completed runs.
    """
    completed = [r for r in results if r.get("status") == "completed"]
    if not completed:
        return None
    best = max(completed, key=lambda r: r["metrics"]["best_test_acc"])
    return {
        "run_name": best["run_name"],
        "best_test_acc": best["metrics"]["best_test_acc"],
    }


def save_json(data: dict, path: Path) -> None:
    """Write *data* to *path* as indented JSON, creating parent dirs as needed.

    Args:
        data: Serialisable dict to write.
        path: Destination file path (created or overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved: {path}")
