
# 3D Object Recognition : Vision & Deep Learning

A two-part pipeline that goes from raw 3D meshes all the way to trained classifiers,
combining **computer vision** (3D reconstruction from images) and **deep learning**
(point cloud classification on ModelNet10/40).

---

## Project structure

```
project/
├── vision/          # 3D reconstruction from Unity captures  →  README_VISION.md
└── deep_learning/   # Point cloud classification (PointNet, DGCNN…)  →  README_DL.md
```

---

## Part 1 — Computer Vision : 3D Reconstruction

> 📄 See **[README_VIC.md]()** for the full documentation.

Reconstructs 3D point clouds from synthetic images rendered in Unity,
using four different methods of increasing complexity.

**Tech stack :** Unity 2021+, Python 3.x, OpenCV, NumPy, Pillow

| Method | Technique                          | Status                                               |
| ------ | ---------------------------------- | ---------------------------------------------------- |
| A      | Stereovision                       | Partial — disparity calibration issues              |
| B      | Canny + Epipolar geometry          | Not working — insufficient keypoints                |
| C      | Multi-view depth maps (36 cameras) | Partial — coordinate frame mismatch Unity → OpenCV |
| D      | Single-view depth map              | Best results — partial reconstruction               |

**Output formats :** `.ply` (MeshLab visualisation) · `.off` (ModelNet10 compatible)

---

## Part 2 — Deep Learning : Point Cloud Classification

> 📄 See **[README_DL.md]()** for the full documentation.

Benchmarks five 3D classification architectures on ModelNet10 and ModelNet40,
with a modular PyTorch codebase for easy model/sampler/dataset swapping.

**Tech stack :** Python 3.12, PyTorch, Open3D, TensorBoard, uv

| Model            | Paper                   | Key idea                                      |
| ---------------- | ----------------------- | --------------------------------------------- |
| SimplePointNet   | —                      | Point-wise MLP + global max pool              |
| PointNet         | Qi et al., CVPR 2017    | Learned spatial alignment (T-Net)             |
| PointNet++       | Qi et al., NeurIPS 2017 | Hierarchical local feature learning           |
| DGCNN            | Wang et al., TOG 2019   | Dynamic k-NN graph (EdgeConv)                 |
| PointTransformer | Zhao et al., ICCV 2021  | Vector self-attention on local neighbourhoods |

**Supported datasets :** ModelNet10 · ModelNet40

---

## End-to-end pipeline

```
Unity scene
    │
    │  Capture (RGB images + depth maps)
    ▼
3D Reconstruction  (vision/)
    │
    │  .off files
    ▼
Point Cloud Classification  (deep_learning/)
    │
    ▼
Predicted class  (chair, table, bathtub…)
```

The `.off` files produced by the vision pipeline are directly compatible
with the deep learning pipeline's dataset format.

---

## Quick links

| Resource                  | Link                                                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| ModelNet10                | [Kaggle](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset)·[Princeton](https://modelnet.cs.princeton.edu/) |
| ModelNet40                | [Kaggle](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset)                                               |
| Unity data (Google Drive) | [Drive folder](https://drive.google.com/drive/folders/1-UUY4vqwnzN8cYLk0dYU7C2SdA2VdurU?usp=sharing)                                    |
| PointNet                  | https://arxiv.org/abs/1612.00593                                                                                                     |
| PointNet++                | https://arxiv.org/abs/1706.02413                                                                                                     |
| DGCNN                     | https://arxiv.org/abs/1801.07829                                                                                                     |
| Point Transformer         | https://arxiv.org/abs/2012.09164                                                                                                     |
