"""Regenerate sequential training plots from existing JSON results.

Scans ``results/sequential/`` for every ``sequential_results.json``, presents
a numbered menu, and re-runs :func:`~src.deep_learning.plotting.plot_sequential_results`
on the selected file(s).

Use this after editing ``plotting.py`` to refresh figures without retraining::

    python -m src.rebuild_figures

The menu lets you pick a single result set or regenerate all at once.
"""

import sys
from pathlib import Path

from src.config import RESULTS_DIR
from src.deep_learning.plotting import plot_sequential_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_results() -> list[Path]:
    """Return all sequential_results.json paths, sorted newest first."""
    search_root = RESULTS_DIR / "sequential"
    if not search_root.exists():
        print(f"No results directory found: {search_root}")
        sys.exit(0)

    found = sorted(
        search_root.rglob("sequential_results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not found:
        print(f"No sequential_results.json files found under: {search_root}")
        sys.exit(0)

    return found


def _menu(results: list[Path]) -> list[Path]:
    """Print a numbered menu and return the user's selection."""
    print("\nAvailable result files (newest first):")
    for i, p in enumerate(results, start=1):
        try:
            rel = p.relative_to(Path.cwd())
        except ValueError:
            rel = p
        print(f"  [{i}] {rel}")

    print(f"\n  [a] Regenerate ALL ({len(results)} file(s))")

    while True:
        raw = input(f"\nSelect (1-{len(results)} or a): ").strip().lower()
        if raw == "a":
            return results
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(results):
                return [results[choice - 1]]
        print(f"  Please enter a number between 1 and {len(results)}, or 'a'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = _find_results()
    selected = _menu(results)

    print()
    for path in selected:
        try:
            rel = path.relative_to(Path.cwd())
        except ValueError:
            rel = path
        print(f"Regenerating: {rel}")
        plot_sequential_results(path)

    print(f"\nDone — {len(selected)} result set(s) regenerated.")
