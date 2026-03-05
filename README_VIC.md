
# Projet Vision par Ordinateur

# Reconstruction 3D depuis ModelNet10

## Prérequis

- Unity 2021+
- Python 3.x
- Packages Python : `opencv-contrib-python`, `numpy`, `Pillow`
- Package Unity : `com.unity.nuget.newtonsoft-json`

---

## Pipeline complet

### Étape 1 : Récupération du dataset

```bash
cd src/builders/utils
python extract_modelnet10_test_dataset.py
```

### Étape 2 : Conversion .off → .obj

```bash
cd src/builders/utils
python conversion_off_obj.py
```

### Étape 3 : Conversion .obj → .prefab

Depuis Unity : `Tools > conversion obj/prefabs`

---

## Méthode A : Stéréovision

### Capture (Unity)

- Script (2 caméras déjà placées dans Unity) : `unity/Visual_V0/Assets/Scripts/ScreenshotCapture.cs`
- Script (2 caméras fixes placées automatiquement): `unity/Visual_V0/Assets/Scripts/ScreenshotCapturePlaceCamera.cs`
- 1 photo par caméra
- Output : `left.png`, `right.png`, `cameras.json` par objet

### Reconstruction (Python)

```bash
cd src/vision/sampling
python reconstruct_stereovision.py
```

- Input : `left.png` + `right.png` + `cameras.json`
- Output : `.ply` (MeshLab) + `.off` (ModelNet10)

### Statut

Résultats partiels, la disparité est mal calibrée pour des objets
synthétiques sans texture

---

## Méthode B : Canny + Géométrie épipolaire

### Capture (Unity)

- Script : `unity/Visual_V0/Assets/Scripts/CameraOrbitCapture.cs`
- 1 caméra mouvante, trajectoire hélicoïdale (36 positions)
- Texture de bruit de Perlin ajoutée sur les objets
- Output : `frame_XXXX.png` + `cameras.json` par objet

### Reconstruction (Python)

```bash
cd src/vision/sampling
python reconstruct.py
```

- Input : images RGB multivues + `cameras.json`
- Output : `.ply` (MeshLab) + `.off` (ModelNet10)

### Statut

Ne fonctionne pas : les surfaces synthétiques sans texture ne
fournissent pas suffisamment de points caractéristiques pour le
matching épipolaire

---

## Méthode C : Depth map multi-vues (36 caméras)

### Capture (Unity)

- Script : `unity/Visual_V0/Assets/Scripts/CameraOrbitCaptureDepthMap.cs`
- 1 caméra mouvante, trajectoire hélicoïdale
- Shader custom de profondeur en niveaux de gris
- Output : `frame_XXXX.png` + `depth_XXXX.png` + `cameras.json` par objet

### Reconstruction (Python)

```bash
cd src/vision/sampling
python reconstruct_from_shaders.py  # sans break
```

- Input : `depth_XXXX.png` (36 fichiers) + `cameras.json`
- Output : `.ply` (MeshLab) + `.off` (ModelNet10)

### Statut

Résultats partiels : les vues ne s'alignent pas correctement
en raison d'une erreur de conversion de repère Unity → OpenCV

---

## Méthode D — Depth map vue unique 

### Capture (Unity)

- Script : `unity/Visual_V0/Assets/Scripts/CameraOrbitCaptureDepthMap.cs`
- Même capture que la Méthode C
- Output : `depth_0000.png` + `cameras.json` par objet

### Reconstruction (Python)

```bash
cd src/vision/sampling
python reconstruct_from_shaders.py  # avec break
```

- Input : `depth_0000.png` (première vue uniquement) + `cameras.json`
- Output : `.ply` (MeshLab) + `.off` (ModelNet10)

### Statut

Meilleurs résultats : forme reconnaissable mais reconstruction
partielle (une seule face visible)

---

## Structure des fichiers de sortie

```
Assets/ModelsDatasetOutput/
└── nom_objet/
    ├── cameras.json       # paramètres caméra
    ├── frame_XXXX.png     # images RGB
    └── depth_XXXX.png     # depth maps

Assets/DatasetTypeModelNet/
└── nom_objet.off          # nuage de points pour ModelNet10
└── nom_objet.ply          # nuage de points pour MeshLab
```

---

## Visualisation

Les fichiers `.ply` peuvent être ouverts dans **MeshLab** pour
visualiser les nuages de points reconstruits.

## Lien avec le projet Deep Learning

Les fichiers `.off` générés sont au format ModelNet10 et peuvent
être utilisés directement comme entrée pour **PointNet++** dans
le pipeline de classification 3D.
