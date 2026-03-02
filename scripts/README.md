# Scripts

Run all scripts from the **project root** with `python -m scripts.<name>`.

| Script | Command | Description |
|---|---|---|
| `sequential_training.py` | `python -m scripts.sequential_training --dataset modelnet10` | Train all models sequentially with curated per-model hyperparameters |
| `grid_training.py` | `python -m scripts.grid_training --dataset modelnet10` | Ablation grid search over model × sampling × n_points × batch_size |
| `visualize_inference.py` | `python -m scripts.visualize_inference` | Interactive Open3D viewer — pick a checkpoint, cycle through test samples with live predictions (SPACE = next, A = quit) |
| `infer_single.py` | `python -m scripts.infer_single` | Run inference on one `.off` file; edit `MODEL_PATH` and `OBJECT_PATH` at the top of the file before running |
| `view_mesh.py` | `python -m scripts.view_mesh` | File-picker dialog to open any `.off` file and inspect its geometry in Open3D |
| `rebuild_figures.py` | `python -m scripts.rebuild_figures` | Regenerate plots from a past `sequential_results.json` without retraining |
| `main.py` | `python -m scripts.main` | Browse all ModelNet10 meshes in Open3D (N / Right = next, P / Left = previous) |
