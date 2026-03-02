# Model Architectures

This document explains the five point cloud classification models implemented in `src/deep_learning/models/`, from simplest to most complex. All models:

- Accept input tensors of shape `(B, N, 3)` — a batch of `B` point clouds, each with `N` points in 3D space.
- Produce logits of shape `(B, num_classes)`.
- Are registered in `ALL_MODELS` in `src/deep_learning/models/__init__.py` and are constructed with `num_classes` inferred dynamically from the dataset.

---

## Summary

| Model | Paper | Key mechanism | Default sampling | Params (≈) |
|---|---|---|---|---|
| SimplePointNet | — | Point-wise MLP + global max pool | Uniform | 1.7 M |
| PointNet | Qi et al., CVPR 2017 | Spatial & feature alignment (T-Nets) | Uniform | 3.5 M |
| PointNet++ | Qi et al., NeurIPS 2017 | Hierarchical Set Abstraction (FPS + ball query) | FPS | 1.7 M |
| DGCNN | Wang et al., TOG 2019 | Dynamic k-NN graph rebuilt per layer (EdgeConv) | Uniform | 1.8 M |
| PointTransformer | Zhao et al., ICCV 2021 | Vector self-attention on local k-NN neighbourhoods | FPS | 2.2 M |

---

## SimplePointNet

**File:** `src/deep_learning/models/SimplePointNet.py`

A minimal baseline with no geometric structure. Useful for verifying the data pipeline and setting a performance floor.

### How it works

Each point is processed independently by a shared MLP (the same weights are applied to every point). A global max-pool then collapses the entire point cloud into a single fixed-size vector, which a classifier MLP turns into class scores.

```
Input (B, N, 3)
    │
    ▼  Shared MLP per point (point-wise)
    │  Linear(3→64) → ReLU → Linear(64→128) → ReLU → Linear(128→1024) → ReLU
    │
    ▼  (B, N, 1024)
    │
    ▼  Global max pool  →  take the highest activation across all N points
    │
    ▼  (B, 1024)  — one descriptor for the whole shape
    │
    ▼  Classifier MLP
    │  Linear(1024→512) → ReLU → Dropout → Linear(512→256) → ReLU → Dropout
    │
    ▼  Linear(256→num_classes)  →  logits (B, C)
```

**Why max-pooling?** It is a symmetric function — the output does not change if you shuffle the input points. This is essential because a point cloud has no natural ordering.

---

## PointNet

**File:** `src/deep_learning/models/PointNet.py`
**Paper:** [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)

Extends SimplePointNet with two *T-Net* modules that learn to align the input into a canonical orientation before feature extraction.

### How it works

**T-Net (Transformation Network)** — A mini-network that predicts a transformation matrix. Applied to the raw 3D coordinates, it rotates and scales the point cloud so that all training examples are in a consistent pose. A second T-Net does the same in 64-d feature space.

```
Input (B, N, 3)
    │
    ▼  Input T-Net: predict 3×3 matrix, apply to all points
    │
    ▼  Shared MLP: Conv(3→64) → Conv(64→64)
    │
    ▼  Feature T-Net: predict 64×64 matrix, apply to all point features
    │  (regularised with orthogonality loss to stay near identity)
    │
    ▼  Shared MLP: Conv(64→64) → Conv(64→128) → Conv(128→1024)
    │
    ▼  (B, N, 1024)
    │
    ▼  Global max pool  →  (B, 1024)
    │
    ▼  Classifier MLP: Linear(1024→512) → Linear(512→256) → Dropout
    │
    ▼  Linear(256→num_classes)  →  logits (B, C)
```

**Key insight:** By learning to align inputs, the model becomes more robust to arbitrary rotations and translations without needing hand-designed data augmentation.

---

## PointNet++

**File:** `src/deep_learning/models/PointNetPP.py`
**Paper:** [arXiv:1706.02413](https://arxiv.org/abs/1706.02413)

Introduces *hierarchical* feature learning. Instead of treating all N points equally, it progressively groups nearby points and extracts local geometry at multiple scales — similar to how a CNN extracts local patterns before combining them into global features.

### How it works

Each **Set Abstraction (SA)** layer:
1. **Farthest Point Sampling (FPS)** — selects well-spread centroid points.
2. **Ball Query** — for each centroid, collects all points within a radius ball.
3. **Local PointNet** — applies a shared MLP inside each ball to extract a local feature.

Three SA layers reduce the point cloud step by step:

```
Input (B, 1024, 3)
    │
    ▼  SA-1: FPS → 512 centroids  |  radius=0.2  |  32 neighbours
    │         MLP [64, 64, 128]
    ▼  (B, 512, 128)   — 512 local descriptors, each summarising a small patch
    │
    ▼  SA-2: FPS → 128 centroids  |  radius=0.4  |  64 neighbours
    │         MLP [128, 128, 256]
    ▼  (B, 128, 256)   — 128 broader regional descriptors
    │
    ▼  SA-3: group ALL remaining points (global)
    │         MLP [256, 512, 1024]
    ▼  (B, 1, 1024)    — single global shape descriptor
    │
    ▼  Flatten  →  (B, 1024)
    │
    ▼  Classifier MLP: FC(1024→512) → BN → ReLU → Dropout
    │                  FC(512→256)  → BN → ReLU → Dropout
    ▼  FC(256→num_classes)  →  logits (B, C)
```

**Key insight:** Local geometry at small scales (e.g., the curve of a chair leg) and large scales (e.g., the overall silhouette) are captured at different SA layers, making the model more discriminative than flat pooling.

---

## DGCNN — Dynamic Graph CNN

**File:** `src/deep_learning/models/DGCNN.py`
**Paper:** [arXiv:1801.07829](https://arxiv.org/abs/1801.07829)

Rather than treating points in isolation, DGCNN constructs a k-nearest-neighbour graph and processes *edges* (relationships between points). Crucially, the graph is **rebuilt at each layer** in the current feature space, not fixed to the original 3D coordinates.

### How it works

**EdgeConv** — for each point, find its k nearest neighbours in the current feature space, form edge features by concatenating the relative and absolute features `[x_j − x_i, x_i]`, then apply a shared MLP and max-pool over the k neighbours.

```
Input (B, N, 3)
    │
    ▼  EdgeConv-1: k-NN in 3D space  →  edge features (B, 6, N)  →  Conv → (B, 64, N)
    │
    ▼  EdgeConv-2: k-NN in 64-d feature space  →  (B, 128, N)  →  (B, 64, N)
    │
    ▼  EdgeConv-3: k-NN in 64-d feature space  →  (B, 128, N)  →  (B, 128, N)
    │
    ▼  EdgeConv-4: k-NN in 128-d feature space  →  (B, 256, N)
    │
    ▼  Concatenate all EdgeConv outputs  →  (B, 512, N)
    │
    ▼  Conv1d(512 → 1024)  →  (B, 1024, N)
    │
    ▼  Dual global pooling: max-pool + avg-pool  →  concatenate  →  (B, 2048)
    │
    ▼  Classifier MLP: Linear(2048→512) → BN → LeakyReLU → Dropout
    │                   Linear(512→256)  → BN → LeakyReLU → Dropout
    ▼  Linear(256→num_classes)  →  logits (B, C)
```

**Key insight:** Rebuilding the graph at every layer means the network defines "neighbourhood" in progressively abstract feature spaces, not just spatial proximity. This lets it capture semantic relationships that may not be geometrically local.

---

## PointTransformer

**File:** `src/deep_learning/models/PointTransformer.py`
**Paper:** [arXiv:2012.09164](https://arxiv.org/abs/2012.09164)

Applies self-attention — the mechanism behind large language models — to point clouds, but with a key adaptation: attention weights are **vectors** (one weight per feature channel) rather than scalars. This allows the model to selectively attend to different feature dimensions for different neighbours.

### How it works

**PointTransformerBlock** — for each point, query its k nearest neighbours. Compute attention weights using the query-key difference *plus* a relative position encoding:

```
attn = softmax( MLP(Q_i − K_j + pos_enc(x_i − x_j)) )    # shape: (k, d)
out_i = Σ_j  attn_j  ⊙  (V_j + pos_enc(x_i − x_j))      # element-wise ⊙
```

Five hierarchical stages, each pairing a PointTransformerBlock with a **TransitionDown** (FPS + kNN aggregation to downsample):

```
Input (B, N, 3)
    │
    ▼  Input MLP: Linear(3→32)
    │
    ▼  Stage 1: Block(32, k=16)  →  TransitionDown(32→64,  pts=256)
    ▼  (B, 256, 64)
    │
    ▼  Stage 2: Block(64, k=16)  →  TransitionDown(64→128, pts=64)
    ▼  (B, 64, 128)
    │
    ▼  Stage 3: Block(128, k=16) →  TransitionDown(128→256, pts=16)
    ▼  (B, 16, 256)
    │
    ▼  Stage 4: Block(256, k=16) →  TransitionDown(256→512, pts=4)
    ▼  (B, 4, 512)
    │
    ▼  Stage 5: Block(512, k=4)
    ▼  (B, 4, 512)
    │
    ▼  Global average pool  →  (B, 512)
    │
    ▼  Classifier MLP: Linear(512→256) → BN → ReLU → Dropout
    ▼  Linear(256→num_classes)  →  logits (B, C)
```

**Key insight:** Vector attention gives each feature channel its own attention score per neighbour, letting the model learn which spatial relationships matter most for each type of feature — a strictly richer aggregation than scalar-attention transformers.
