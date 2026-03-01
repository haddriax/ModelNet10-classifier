# ModelNet Classifier

A modular PyTorch pipeline for benchmarking 3D point cloud classification models on the ModelNet10 and ModelNet40 datasets.

---

## Why this project?

After working on 2D classification (CIFAR), I was curious about 3D classification from point clouds, the kind of input a LiDAR sensor produces. This project gave me a practical playground to:

- Build a modular codebase where swapping models, samplers, or datasets is a one-line change.
- Implement five architectures from their original papers and understand the design choices behind each.
- Practice optimizer and scheduler tuning, and use TensorBoard to compare runs in real time.

---

## Quick start

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv), CUDA 12.6 (for GPU training).

```bash
# Install all dependencies
uv sync

# Interactive 3D mesh viewer  (N / Right = next mesh, P / Left = previous)
python -m src.main

# Train all models sequentially on ModelNet10 (default) or ModelNet40
python -m src.sequential_training --dataset modelnet10
python -m src.sequential_training --dataset modelnet40

# Full ablation grid  (model × sampling × n_points × batch_size)
python -m src.grid_training

# Inference visualiser  (interactive menu → pick checkpoint → 3D viewer with live predictions)
python -m src.visualize_inference

# Monitor training in real time
tensorboard --logdir=runs
```

---

## Architecture overview

The pipeline is a straight line from raw mesh file to trained classifier:

```
data/ModelNet*/models/
      │  .off files
      ▼
 Mesh3DBuilder          parse OFF → Mesh3D (Open3D wrapper)
      │
      ▼
 PointCloudDataset      sample N points from surface → normalise to unit sphere
      │  (B, N, 3) tensors
      ▼
 ModelTrainer           Adam / custom optimiser + LR scheduler + early stopping
      │                 TensorBoard logging  ·  checkpoint saving
      ▼
results/sequential/{dataset}/{timestamp}/
      sequential_results.json  +  comparison plots (.png)
```

For the full module breakdown see [MODULES.md](MODULES.md).
For model architecture details see [MODELS.md](MODELS.md).

---

## Datasets

| Dataset    | Classes | Train meshes | Test meshes | Download                                                                                                                                    |
|------------|---------|--------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| ModelNet10 | 10      | ~3 991       | ~908        | [Kaggle](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset) . [Princeton](https://modelnet.cs.princeton.edu/) |
| ModelNet40 | 40      | ~9 843       | ~2 468      | [Kaggle](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset) . [Princeton](https://modelnet.cs.princeton.edu/) |

Place datasets under `data/` with this layout (matching the Princeton zip structure):

```
data/
├── ModelNet10/models/{class}/train/*.off
│                            /test/*.off
└── ModelNet40/models/{class}/train/*.off
                             /test/*.off
```

### OFF file format

Meshes are stored in the OFF (Object File Format) — a plain-text polygon mesh:

```
OFF              ← header (ModelNet40 writes this as "OFF3514 3546 0" on one line)
3514 3546 0      ← num_vertices  num_faces  num_edges
x y z            ← one vertex per line
...
3 v0 v1 v2       ← one face per line  (first token = vertex count)
...
```

Both header variants (standard two-line and compact one-line) are handled automatically by the parser.

---

## Sampling

Point clouds are sampled from mesh surfaces before being fed to the network. All models use **1 024 points** by default.

| Method | Description | Speed |
|---|---|---|
| `uniform` | Uniform random surface sampling (Open3D) | Fast |
| `fps` | Farthest Point Sampling — maximises spatial coverage | Slower |
| `poisson` | Poisson disk sampling — evenly spaced | Slower |

**Train vs test:** The training set is re-sampled every epoch (implicit augmentation). The test set is sampled once and cached to disk so evaluation is reproducible across runs.
Cache location: `data/{Dataset}/cache/pointcloud_{split}_{n}pts_{method}/`

All sampled point clouds are **normalised to the unit sphere** (centred at the origin, maximum radius = 1) before being passed to the model.

---

## Models

| Model | Paper | Key idea | Default sampling |
|---|---|---|---|
| SimplePointNet | — | Point-wise MLP + global max pool (baseline) | Uniform |
| PointNet | Qi et al., CVPR 2017 | Spatial & feature alignment via learned T-Nets | Uniform |
| PointNet++ | Qi et al., NeurIPS 2017 | Hierarchical local feature learning (FPS + ball query) | FPS |
| DGCNN | Wang et al., TOG 2019 | Dynamic k-NN graph rebuilt at each layer (EdgeConv) | Uniform |
| PointTransformer | Zhao et al., ICCV 2021 | Vector self-attention on local neighbourhoods | FPS |

See [MODELS.md](MODELS.md) for architecture diagrams and simplified explanations.

---

## Inference visualiser

`src/visualize_inference.py` loads any trained checkpoint and lets you browse test samples in an interactive Open3D window while watching live predictions in the terminal.

### Launch

```bash
python -m src.visualize_inference
```

No arguments are needed. The script automatically scans `models/` for every `.pth` file and presents a numbered menu:

```
Available checkpoints:
  [1] models/sequential/modelnet10/2026-03-01_120000/PointNet_uniform_pts1024_bs32.pth
  [2] models/sequential/modelnet40/2026-03-01_165413/DGCNN_uniform_pts1024_bs32.pth
  ...

Select checkpoint (1-2):
```

### What happens next

| Step | What the script does |
|---|---|
| Config parsing | Reads model name, sampling method and point count from the filename automatically |
| Dataset detection | Reads `modelnet10` / `modelnet40` from the checkpoint path → loads the matching test set and sets `num_classes` (10 or 40) |
| Manual fallback | If detection fails (e.g. a checkpoint in a non-standard location), the script prompts you to type the dataset and model parameters |
| Model loading | Instantiates the correct architecture with the right class count and loads weights |

### Interactive viewer

Once the window opens, each sample shows:
- **Gray wireframe** — the raw triangle mesh from the OFF file
- **Green point cloud** — the N points sampled from the surface (exactly what the model sees, after unit-sphere normalisation)

The terminal prints for every sample:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sample 1/908  |  chair_0890.off
Ground truth : chair
Predicted    : chair  ✓  (confidence: 94.3%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Controls:**

| Key | Action |
|---|---|
| `Space` | Next random test sample |
| `A` | Quit |
| `Escape` | Quit |

---

## Results

Sequential benchmark on **ModelNet40** (40 classes, 1 024 points, batch size 32):

![Sequential model comparison on ModelNet40](images/sequential_model_comparison.png)

Results and plots are saved per run under `results/sequential/{dataset}/{timestamp}/` — older runs are never overwritten. Model checkpoints are saved under `models/sequential/{dataset}/{timestamp}/`.

---

## Project structure

```
src/
├── config.py                   # Global path constants
├── main.py                     # Interactive mesh viewer
├── sequential_training.py      # Entry point: sequential benchmark
├── grid_training.py            # Entry point: ablation grid
├── visualize_inference.py      # Entry point: inference visualiser
├── geometry/
│   ├── Mesh_3D.py              # Open3D mesh wrapper + point sampling
│   └── sampling.py             # Sampling enum (UNIFORM, FPS, POISSON)
├── builders/
│   ├── mesh_3D_builder.py      # OFF file → Mesh3D
│   └── utils/format_parser.py  # OFF format parser (standard + compact)
├── dataset/
│   ├── base_modelnet_dataset.py
│   └── point_cloud_dataset.py  # Caching + normalisation
└── deep_learning/
    ├── configs.py              # ModelConfig dataclass
    ├── model_trainer.py        # Training loop + TensorBoard + checkpoints
    ├── sequential_trainer.py   # run_sequential() library function
    ├── grid_search.py          # GridSearch + GridSearchConfig
    ├── plotting.py             # Comparison plots
    └── models/
        ├── SimplePointNet.py
        ├── PointNet.py
        ├── PointNetPP.py
        ├── DGCNN.py
        └── PointTransformer.py
```
---

## References

| Topic | Link |
|---|---|
| PointNet | https://arxiv.org/abs/1612.00593 |
| PointNet++ | https://arxiv.org/abs/1706.02413 |
| DGCNN | https://arxiv.org/abs/1801.07829 |
| Point Transformer | https://arxiv.org/abs/2012.09164 |
| Open3D | https://www.open3d.org/ |
| ModelNet | https://modelnet.cs.princeton.edu/ |
