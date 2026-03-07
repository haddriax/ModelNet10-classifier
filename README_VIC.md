
# Computer Vision Project

# 3D Reconstruction from ModelNet10

## Prerequisites

* Unity 2021+
* Python 3.x
* Python packages: `opencv-contrib-python`, `numpy`, `Pillow`
* Unity package: `com.unity.nuget.newtonsoft-json`

## Data

### Google Drive link:

https://drive.google.com/drive/folders/1-UUY4vqwnzN8cYLk0dYU7C2SdA2VdurU?usp=sharing

### Available folders:

* unity/Visual_V0/Assets/ScreenShots
* unity/Visual_V0/Assets/ModelsDatasetOutput
* unity/Visual_V0/Assets/DatasetTypeModelNet
* unity/Visual_V0/Assets/DatasetTypeModelNetMultiView
* unity/Visual_V0/Assets/DatasetTypeModelNetStereovision
* unity/Visual_V0/Assets/DatasetTypeModelNetCanny
* unity/Visual_V0/Assets/Resources

---

## Full pipeline

### Step 1: Dataset retrieval

```bash
cd src/builders/utils
python extract_modelnet10_test_dataset.py
```

### Step 2: .off → .obj conversion

```bash
cd src/builders/utils
python conversion_off_obj.py
```

### Step 3: .obj → .prefab conversion

From Unity: `Tools > conversion obj/prefabs`

---

## Method A: Stereovision

### Capture (Unity)

* Script (2 cameras already placed in Unity): `unity/Visual_V0/Assets/Scripts/ScreenshotCapture.cs`
* Script (2 fixed cameras placed automatically): `unity/Visual_V0/Assets/Scripts/ScreenshotCapturePlaceCamera.cs`
* 1 photo per camera
* Output: `left.png`, `right.png`, `cameras.json` per object

### Reconstruction (Python)

```bash
cd src/vision/sampling
python reconstruct_stereovision.py
```

* Input: `left.png` + `right.png` + `cameras.json`
* Output: `.ply` (MeshLab) + `.off` (ModelNet10)

### Status

Partial results — disparity is poorly calibrated for synthetic
objects without texture.

---

## Method B: Canny + Epipolar Geometry

### Capture (Unity)

* Script: `unity/Visual_V0/Assets/Scripts/CameraOrbitCapture.cs`
* 1 moving camera, helical trajectory (36 positions)
* Perlin noise texture added on objects
* Output: `frame_XXXX.png` + `cameras.json` per object

### Reconstruction (Python)

```bash
cd src/vision/sampling
python reconstruct.py
```

* Input: multi-view RGB images + `cameras.json`
* Output: `.ply` (MeshLab) + `.off` (ModelNet10)

### Status

Not working — synthetic surfaces without texture do not provide
enough keypoints for epipolar matching.

---

## Method C: Multi-view Depth Maps (36 cameras)

### Capture (Unity)

* Script: `unity/Visual_V0/Assets/Scripts/CameraOrbitCaptureDepthMap.cs`
* 1 moving camera, helical trajectory
* Custom grayscale depth shader
* Output: `frame_XXXX.png` + `depth_XXXX.png` + `cameras.json` per object

### Reconstruction (Python)

```bash
cd src/vision/sampling
python reconstruct_from_shaders.py  # without break
```

* Input: `depth_XXXX.png` (36 files) + `cameras.json`
* Output: `.ply` (MeshLab) + `.off` (ModelNet10)

### Status

Partial results : views do not align correctly due to a coordinate
frame conversion error Unity → OpenCV.

---

## Method D: Single-view Depth Map

### Capture (Unity)

* Script: `unity/Visual_V0/Assets/Scripts/CameraOrbitCaptureDepthMap.cs`
* Same capture as Method C
* Output: `depth_0000.png` + `cameras.json` per object

### Reconstruction (Python)

```bash
cd src/vision/sampling
python reconstruct_from_shaders.py  # with break
```

* Input: `depth_0000.png` (first view only) + `cameras.json`
* Output: `.ply` (MeshLab) + `.off` (ModelNet10)

### Status

Best results : recognisable shape but partial reconstruction
(only one face visible).

---

## Output file structure

```
Assets/ModelsDatasetOutput/
└── object_name/
    ├── cameras.json       # camera parameters
    ├── frame_XXXX.png     # RGB images
    └── depth_XXXX.png     # depth maps

Assets/DatasetTypeModelNet/
└── object_name.off        # point cloud for ModelNet10
└── object_name.ply        # point cloud for MeshLab
```

---

## Visualisation

`.ply` files can be opened in **MeshLab** to visualise the
reconstructed point clouds.

## Link with the Deep Learning project

The generated `.off` files are in ModelNet10 format and can be used
directly as input for **PointNet++** in the 3D classification pipeline.
