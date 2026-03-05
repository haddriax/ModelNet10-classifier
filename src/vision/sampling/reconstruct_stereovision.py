import cv2
import numpy as np
import json
import os
from pathlib import Path


def load_stereo_params(json_path):
    """Charge les paramètres des 2 caméras depuis cameras.json"""
    with open(json_path) as f:
        data = json.load(f)

    cameras = {}
    for cam in data:
        fov  = cam["fov"]
        w, h = cam["width"], cam["height"]
        fx   = (w / 2) / np.tan(np.radians(fov / 2))
        K    = np.array([
            [fx,  0, w/2],
            [ 0, fx, h/2],
            [ 0,  0,   1]
        ], dtype=np.float64)

        # view_matrix Unity → R, t dans repère OpenCV
        vm = np.array(cam["view_matrix"]).reshape(4, 4, order='F')
        vm[2, :] *= -1  # correction Unity LH → RH
        R = vm[:3, :3]
        t = vm[:3,  3]

        cameras[cam["name"]] = {
            "K": K, "R": R, "t": t,
            "width": w, "height": h,
            "far_clip":  cam.get("far_clip",  100.0),
            "near_clip": cam.get("near_clip", 0.01),
        }
    return cameras["left"], cameras["right"]


def rectify_stereo(cam_l, cam_r, img_l, img_r):
    """
    Rectification stéréo : redresse les 2 images pour que les lignes
    épipolaires soient horizontales (nécessaire pour StereoSGBM).
    """
    h, w = img_l.shape[:2]

    # Rotation et translation relatives entre les 2 caméras
    R_rel = cam_r["R"] @ cam_l["R"].T
    t_rel = cam_r["t"] - R_rel @ cam_l["t"]

    # Rectification via OpenCV
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cam_l["K"], None,   # pas de distorsion (scène synthétique)
        cam_r["K"], None,
        (w, h),
        R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    # Calcul des maps de remapping
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        cam_l["K"], None, R1, P1, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        cam_r["K"], None, R2, P2, (w, h), cv2.CV_32FC1)

    # Application des maps
    rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

    return rect_l, rect_r, Q


def compute_disparity(rect_l, rect_r):
    """
    Calcule la carte de disparité avec StereoSGBM.
    La disparité encode le décalage horizontal entre les 2 images :
    plus un objet est proche, plus la disparité est grande.
    """
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    min_disp  = 0
    num_disp  = 128   # doit être multiple de 16
    block_size = 5

    stereo = cv2.StereoSGBM_create(
        minDisparity    = min_disp,
        numDisparities  = num_disp,
        blockSize       = block_size,
        P1              = 8  * 3 * block_size ** 2,
        P2              = 32 * 3 * block_size ** 2,
        disp12MaxDiff   = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange    = 32,
        mode            = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    return disparity


def disparity_to_pointcloud(disparity, Q, rect_l, cam_l):
    """
    Rétroprojection disparité → nuage de points 3D via la matrice Q.
    Q est calculée par stereoRectify et encode la géométrie stéréo.
    """
    # Masque : ignore les pixels sans disparité valide
    mask = disparity > 0

    # Rétroprojection 3D via Q
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    # Récupère couleur depuis image gauche rectifiée
    colors = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)

    pts   = points_3d[mask]
    cols  = colors[mask]

    # Filtre les points aberrants
    dist = np.linalg.norm(pts, axis=1)
    valid = (dist > cam_l["near_clip"]) & (dist < cam_l["far_clip"] * 0.5)
    pts  = pts[valid]
    cols = cols[valid]

    # Filtre statistique centré sur la médiane
    center    = np.median(pts, axis=0)
    dist_cent = np.linalg.norm(pts - center, axis=1)
    mask_stat = dist_cent < np.percentile(dist_cent, 95)
    pts  = pts[mask_stat]
    cols = cols[mask_stat]

    print(f"  Nuage de points : {len(pts)} points")
    return pts, cols


def save_ply(points, colors, path):
    """Sauvegarde nuage de points coloré en .ply"""
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def save_off(points, path):
    """Sauvegarde nuage de points en .off (format ModelNet10)"""
    with open(path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(points)} 0 0\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def reconstruct_stereo(folder_path, output_dir):
    """Pipeline complet de reconstruction stéréo pour un objet."""
    folder = Path(folder_path)
    name   = folder.name
    print(f"\n=== Reconstruction stéréo : {name} ===")

    # Chargement
    json_path = folder / "cameras.json"
    img_l_path = folder / "left.png"
    img_r_path = folder / "right.png"

    if not json_path.exists() or not img_l_path.exists() or not img_r_path.exists():
        print(f"  [SKIP] fichiers manquants")
        return

    cam_l, cam_r = load_stereo_params(str(json_path))
    img_l = cv2.imread(str(img_l_path))
    img_r = cv2.imread(str(img_r_path))

    if img_l is None or img_r is None:
        print(f"  [SKIP] images illisibles")
        return

    # Pipeline
    rect_l, rect_r, Q = rectify_stereo(cam_l, cam_r, img_l, img_r)
    disparity          = compute_disparity(rect_l, rect_r)
    pts, cols          = disparity_to_pointcloud(disparity, Q, rect_l, cam_l)

    if len(pts) == 0:
        print(f"  ERREUR : aucun point reconstruit")
        return

    # Sauvegarde
    os.makedirs(output_dir, exist_ok=True)
    save_ply(pts, cols, os.path.join(output_dir, f"{name}.ply"))
    save_off(pts,       os.path.join(output_dir, f"{name}.off"))
    print(f"  Sauvegardé → {output_dir}/{name}.off")


