# Modules Reference

This document describes every package and module in `src/`, in pipeline order — from raw mesh files on disk to trained model and visualisation.

---

## Pipeline at a glance

```
.off files on disk
    └─► src/builders/          parse mesh
            └─► src/geometry/  represent + sample point cloud
                    └─► src/dataset/       batch + cache
                            └─► src/deep_learning/   train + evaluate
                                        └─► results/ + models/
```

---

## `src/config.py`

Central location for all project-wide path constants. Import these instead of hard-coding paths anywhere else.

| Constant | Resolved path |
|---|---|
| `PROJECT_ROOT` | Repository root |
| `DATA_DIR` | `data/ModelNet10/models/` |
| `MODELNET40_DIR` | `data/ModelNet40/models/` |
| `MODELS_DIR` | `models/` |
| `RESULTS_DIR` | `results/` |

---

## `src/geometry/`

Low-level 3D geometry primitives.

### `Sampling` (enum) — `sampling.py`

Three point-cloud sampling strategies, passed wherever a sampling method is required:

| Value | Open3D method | Behaviour |
|---|---|---|
| `UNIFORM` | `sample_points_uniformly` | Fast random surface samples |
| `FARTHEST_POINT` | `farthest_point_down_sample` | Greedy max-coverage (FPS) |
| `POISSON` | `sample_points_poisson_disk` | Evenly-spaced disk sampling |

### `Mesh3D` — `Mesh_3D.py`

Thin wrapper around an Open3D `TriangleMesh`.

```python
mesh = Mesh3D(vertices, faces)           # np.ndarray (V,3), (F,3)
pts  = mesh.sample_points(n, method)     # → np.ndarray (n, 3), float32
```

`sample_points` caches the result in memory keyed on `(n_points, method)`. Pass `force_resample=True` to discard the cache (used when building the training set disk cache).

---

## `src/builders/`

Converts raw `.off` files into `Mesh3D` objects.

### `OffMeshParser` — `utils/format_parser.py`

```python
vertices, faces = OffMeshParser.parse_off(lines)
```

Handles both OFF header variants:
- **Standard** (ModelNet10): `"OFF\n3514 3546 0\n…"`
- **Compact** (ModelNet40): `"OFF3514 3546 0\n…"`

### `Mesh3DBuilder` — `mesh_3D_builder.py`

```python
mesh = Mesh3DBuilder.from_off_file(path)   # Path → Mesh3D
```

Reads the file, delegates parsing to `OffMeshParser`, and wraps the result in a `Mesh3D`.

---

## `src/dataset/`

PyTorch `Dataset` implementations with split management and disk caching.

### `BaseModelNetDataset` — `base_modelnet_dataset.py`

Abstract base class. Handles:
- **Class discovery** — scans `root_dir/` for subdirectories; builds `class_to_idx` and `idx_to_class` mappings dynamically (no hardcoded class list).
- **Split strategy** — `use_existing_split=True` reads `{class}/train/` and `{class}/test/` folders; `use_existing_split=False` creates a random train/test split from all files.
- **Caching hook** — calls `_build_cache()` if `cache_processed=True`; subclasses override this.
- **`__getitem__`** — returns `(tensor, label_int)` and applies an optional `transform`.

### `PointCloudDataset` — `point_cloud_dataset.py`

Concrete subclass. Adds:
- **Disk cache** at `data/{dataset}/cache/pointcloud_{split}_{n}pts_{method}/` — one `.npy` file per mesh. Reused automatically if count matches.
- **Train/test asymmetry** — test set is always cached (`cache_processed=True` by default); training set uses dynamic re-sampling each call for implicit data augmentation.
- **Unit-sphere normalisation** — every point cloud is centred at the origin and scaled so `max_norm = 1` before being returned. Applied in-memory; the on-disk `.npy` files retain raw coordinates.

```python
ds = PointCloudDataset(
    root_dir=DATA_DIR,
    split='train',
    n_points=1024,
    sampling_method=Sampling.FARTHEST_POINT,
)
points, label = ds[0]   # torch.Tensor (1024, 3), int
```

### `dataset_factory` — `deep_learning/dataset_factory.py`

```python
train_ds, test_ds = make_datasets(n_points, sampling_method, data_dir)
```

Convenience function used by the sequential trainer to avoid boilerplate.

---

## `src/deep_learning/configs.py`

### `ModelConfig` (dataclass)

Per-model configuration passed to `run_sequential()`. Only `sampling` is required; all other fields fall back to the global values supplied to `run_sequential()` when set to `None`.

```python
ModelConfig(
    sampling="fps",           # required: "uniform" | "fps" | "poisson"
    lr=0.001,                 # optional override
    patience=25,              # optional override (epochs without improvement)
    early_stop_metric="f1",   # optional override: "accuracy" | "f1" | "loss"
    epochs=200,               # optional override
    optimizer_factory=lambda params, lr: AdamW(params, lr=lr, weight_decay=1e-4),
    scheduler_factory=lambda opt, _: StepLR(opt, step_size=20, gamma=0.7),
)
```

`optimizer_factory` signature: `(parameters, lr) → Optimizer`.
`scheduler_factory` signature: `(optimizer, epochs_remaining) → LRScheduler`.

---

## `src/deep_learning/model_trainer.py`

### `ModelTrainer`

The core training loop. Accepts a model, two datasets, and configuration; returns a `TrainingResults` TypedDict.

Key behaviours:
- **Optimiser** — Adam with auto LR `0.001 × (batch_size / 32)` by default; override with `optimizer_factory`.
- **Scheduler** — `CosineAnnealingLR` by default; override with `scheduler_factory`.
- **Early stopping** — monitors `accuracy`, `f1` (macro), or `loss`; stops after `patience` epochs without improvement.
- **TensorBoard** — logs train/test loss, accuracy, per-class precision/recall/F1, and learning rate to `runs/{experiment_name}/`.
- **Checkpoints** — saves `{name}.pth` (latest) and `{name}_best.pth` (best metric) to the configured `save_model` path.

```python
trainer = ModelTrainer(train_dataset, test_dataset, model, save_model=path, ...)
results = trainer.train(epochs=200)   # → TrainingResults TypedDict
```

---

## `src/deep_learning/sequential_trainer.py`

### `run_sequential(configs, *, n_points, batch_size, ..., data_dir, results_dir, models_dir)`

Trains every model in `configs` one after another. Key details:
- **Dataset cache** — datasets are shared across models that use the same `(n_points, sampling)` pair, avoiding redundant re-building.
- **`num_classes` is dynamic** — inferred as `len(train_ds.class_to_idx)` after dataset construction; no hardcoding.
- **Output** — saves `sequential_results.json` and four comparison plots (model comparison, per-class accuracy, per-class F1, training efficiency) to `results_dir`.
- **Checkpoint paths** — `models_dir / f"{run_name}.pth"`.

The entry point `src/sequential_training.py` builds timestamped paths:
```
results/sequential/{dataset}/YYYY-MM-DD_HHMMSS/
models/sequential/{dataset}/YYYY-MM-DD_HHMMSS/
```

---

## `src/deep_learning/grid_search.py`

### `GridSearchConfig`

Defines the Cartesian product to explore:

```python
GridSearchConfig(
    model_classes=[PointNet, DGCNN],
    sampling_methods=[Sampling.UNIFORM, Sampling.FARTHEST_POINT],
    n_points_list=[512, 1024],
    batch_sizes=[16, 32],
)
```

### `GridSearch`

Iterates every combination, caches datasets by `(n_points, sampling)`, and saves results after each run for crash recovery. Entry point: `python -m src.grid_training`.

---

## `src/deep_learning/plotting.py`

| Function | Output files | Used by |
|---|---|---|
| `plot_sequential_results(results_path)` | `sequential_model_comparison.png`, `sequential_per_class_accuracy.png`, `sequential_per_class_f1.png`, `sequential_training_efficiency.png` | `run_sequential()`, `rebuild_figures.py` |
| `plot_training_efficiency(runs, output_dir, plt)` | `sequential_training_efficiency.png` | called by `plot_sequential_results()` |
| `create_ablation_plots(results_path)` | `accuracy_comparison.png`, `npoints_effect.png`, `batchsize_effect.png`, `sampling_comparison.png`, `model_heatmap.png` | `GridSearch` |

All plots are written to the same directory as the JSON file, or to `output_dir` if provided.

---

## Entry points

| Script | Invocation | Purpose |
|---|---|---|
| `src/main.py` | `python -m src.main` | Interactive Open3D viewer — browse all meshes (N/Right = next, P/Left = prev) |
| `src/sequential_training.py` | `python -m src.sequential_training [--dataset modelnet10\|modelnet40]` | Train all models sequentially with curated hyperparameters |
| `src/grid_training.py` | `python -m src.grid_training` | Full ablation over model × sampling × n_points × batch_size |
| `src/visualize_inference.py` | `python -m src.visualize_inference` | Load a trained checkpoint, run inference on test meshes, visualise in 3D |
| `src/rebuild_figures.py` | `python -m src.rebuild_figures` | Re-run `plot_sequential_results()` on any past JSON without retraining — useful after editing `plotting.py` |
