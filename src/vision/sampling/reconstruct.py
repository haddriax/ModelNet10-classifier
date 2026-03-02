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
        fy = fx
        K = np.array([
            [fx,  0, w/2],
            [ 0, fy, h/2],
            [ 0,  0,   1]
        ], dtype=np.float64)

        vm = np.array(cam["view_matrix"]).reshape(4, 4, order='F')
        vm[1, :] *= -1
        vm[2, :] *= -1
        R = vm[:3, :3]
        t = vm[:3, 3]

        cameras.append({
            "K": K,
            "R": R,
            "t": t,
            "image_name": cam["image_name"]
        })
    return cameras


def extract_and_match(img1, img2):
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def triangulate_pair(cam1, cam2, pts1, pts2):
    P1 = cam1["K"] @ np.hstack([cam1["R"], cam1["t"].reshape(3,1)])
    P2 = cam2["K"] @ np.hstack([cam2["R"], cam2["t"].reshape(3,1)])

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T  

    dist = np.linalg.norm(pts3d, axis=1)
    mask = dist < np.percentile(dist, 95)
    return pts3d[mask]


def reconstruct(folder_path):
    folder = Path(folder_path)
    cameras = load_camera_params(folder / "cameras.json")

    all_points = []

    pairs = []
    n = len(cameras)
    for i in range(n):
        pairs.append((i, (i+1) % n))        
        pairs.append((i, (i + n//4) % n))  

    for i, j in pairs:
        cam1, cam2 = cameras[i], cameras[j]
        img1 = cv2.imread(str(folder / cam1["image_name"]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(folder / cam2["image_name"]), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue

        pts1, pts2 = extract_and_match(img1, img2)
        if len(pts1) < 8:
            continue

        pts3d = triangulate_pair(cam1, cam2, pts1, pts2)
        all_points.append(pts3d)

    if not all_points:
        print(f"ERREUR : aucun point reconstruit pour {folder_path}")
        return None

    cloud = np.vstack(all_points)
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
    """Format .off pour ModelNet10"""
    with open(path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(points)} 0 0\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def reconstruct_folder(folder_path, output_dir):
    name = Path(folder_path).name
    cloud = reconstruct(folder_path)
    if cloud is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    save_ply(cloud, os.path.join(output_dir, f"{name}.ply"))
    save_off(cloud, os.path.join(output_dir, f"{name}.off"))
