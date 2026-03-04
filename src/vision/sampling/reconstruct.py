import cv2
import numpy as np
import json
import os
from pathlib import Path


def load_camera_params(json_path):
    with open(json_path) as f:
        data = json.load(f)
    cameras = []
    for cam in data:
        fov = cam["fov"]
        w, h = cam["width"], cam["height"]
        fx = (w / 2) / np.tan(np.radians(fov / 2))
        K = np.array([
            [fx,  0, w/2],
            [ 0, fx, h/2],
            [ 0,  0,   1]
        ], dtype=np.float64)

        vm = np.array(cam["view_matrix"]).reshape(4, 4, order='F')
        vm[2, :] *= -1
        R = vm[:3, :3]
        t = vm[:3, 3]

        cameras.append({"K": K, "R": R, "t": t, "image_name": cam["image_name"]})
    return cameras


def extract_contour_points(img1, img2):
    _, mask1 = cv2.threshold(img1, 10, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    edges1 = cv2.Canny(mask1, 50, 150)
    edges2 = cv2.Canny(mask2, 50, 150)
    pts1 = np.argwhere(edges1 > 0)[:, ::-1].astype(np.float32)
    pts2 = np.argwhere(edges2 > 0)[:, ::-1].astype(np.float32)
    return pts1, pts2, mask1, mask2


def match_with_epipolar(pts1, pts2, K1, K2, R1, t1, R2, t2):
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    tx = np.array([
        [0,        -t_rel[2],  t_rel[1]],
        [t_rel[2],  0,        -t_rel[0]],
        [-t_rel[1], t_rel[0],  0       ]
    ])
    E = tx @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    matched1, matched2 = [], []

    for pt in pts1[::10]:  # sous-échantillonnage plus agressif
        pt_h = np.array([pt[0], pt[1], 1.0])
        line = F @ pt_h
        a, b, c = line
        denom = np.sqrt(a**2 + b**2)
        if denom < 1e-8:
            continue

        dists = np.abs(a * pts2[:, 0] + b * pts2[:, 1] + c) / denom
        idx = np.argmin(dists)

        if dists[idx] > 1.5:  # seuil plus strict
            continue

        # Vérification symétrique : le match retour doit pointer vers pt
        pt2 = pts2[idx]
        pt2_h = np.array([pt2[0], pt2[1], 1.0])
        line_back = F.T @ pt2_h
        a2, b2, c2 = line_back
        denom2 = np.sqrt(a2**2 + b2**2)
        if denom2 < 1e-8:
            continue
        dist_back = abs(a2 * pt[0] + b2 * pt[1] + c2) / denom2

        if dist_back < 1.5:  # cohérent dans les deux sens
            matched1.append(pt)
            matched2.append(pt2)

    if len(matched1) < 8:
        return np.array([]), np.array([])

    m1 = np.array(matched1, dtype=np.float32)
    m2 = np.array(matched2, dtype=np.float32)

    # RANSAC pour éliminer les outliers restants
    _, mask = cv2.findFundamentalMat(m1, m2, cv2.FM_RANSAC, 1.0, 0.99)
    if mask is None:
        return np.array([]), np.array([])

    mask = mask.ravel().astype(bool)
    return m1[mask], m2[mask]


def triangulate_pair(cam1, cam2, pts1, pts2):
    if len(pts1) < 8:
        return np.array([])

    P1 = cam1["K"] @ np.hstack([cam1["R"], cam1["t"].reshape(3, 1)])
    P2 = cam2["K"] @ np.hstack([cam2["R"], cam2["t"].reshape(3, 1)])

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    w = pts4d[3]
    mask_w = np.abs(w) > 1e-6
    pts3d = (pts4d[:3, mask_w] / w[mask_w]).T

    # Filtre points devant les deux caméras (z > 0 dans chaque repère)
    def in_front(R, t, pts):
        pts_cam = (R @ pts.T + t.reshape(3, 1))
        return pts_cam[2] > 0

    mask = in_front(cam1["R"], cam1["t"], pts3d) & in_front(cam2["R"], cam2["t"], pts3d)
    pts3d = pts3d[mask]

    if len(pts3d) == 0:
        return np.array([])

    # Filtre statistique (retire les points trop loin)
    dist = np.linalg.norm(pts3d, axis=1)
    mask2 = dist < np.percentile(dist, 90)
    return pts3d[mask2]

def reconstruct(folder_path):
    folder = Path(folder_path)
    cameras = load_camera_params(folder / "cameras.json")
    n = len(cameras)
    all_points = []

    pairs = []
    for i in range(n):
        pairs.append((i, (i + 1) % n))
        pairs.append((i, (i + n // 4) % n))

    for i, j in pairs:
        if i == j:
            continue
        cam1, cam2 = cameras[i], cameras[j]
        img1 = cv2.imread(str(folder / cam1["image_name"]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(folder / cam2["image_name"]), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue

        pts1, pts2, mask1, mask2 = extract_contour_points(img1, img2)
        if len(pts1) < 8 or len(pts2) < 8:
            continue

        pts1, pts2 = match_with_epipolar(
            pts1, pts2,
            cam1["K"], cam2["K"],
            cam1["R"], cam1["t"],
            cam2["R"], cam2["t"]
        )
        if len(pts1) < 8:
            continue

        pts3d = triangulate_pair(cam1, cam2, pts1, pts2)
        if len(pts3d) > 0:
            all_points.append(pts3d)
            print(f"  Paire ({i},{j}) → {len(pts3d)} points")

    if not all_points:
        print(f"ERREUR : aucun point reconstruit pour {folder_path}")
        return None

    cloud = np.vstack(all_points)
    print(f"  Total : {len(cloud)} points")
    return cloud


def save_ply(points, path):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def save_off(points, path):
    with open(path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(points)} 0 0\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def reconstruct_folder(folder_path, output_dir):
    name = Path(folder_path).name
    print(f"\n=== Reconstruction : {name} ===")
    cloud = reconstruct(folder_path)
    if cloud is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    save_ply(cloud, os.path.join(output_dir, f"{name}.ply"))
    save_off(cloud, os.path.join(output_dir, f"{name}.off"))
    print(f"  Sauvegardé → {output_dir}/{name}.off")