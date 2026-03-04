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

        pos = np.array(cam["position"], dtype=np.float64) 

        c2w = np.array(cam["cam_to_world"]).reshape(4, 4, order='F')

        c2w[1, :] *= -1
        c2w[2, :] *= -1

        cameras.append({
            "K": K,
            "c2w": c2w,
            "pos": pos,  
            "image_name": cam["image_name"],
            "depth_name": cam.get("depth_name", cam["image_name"].replace("frame_", "depth_")),
            "far_clip": cam.get("far_clip", 100.0),
            "near_clip": cam.get("near_clip", 0.01),
            "width": w,
            "height": h
        })
    return cameras

def decode_depth(depth_img, far_clip):
    d = depth_img[:, :, 0].astype(np.float32) / 255.0
    return d * far_clip


def depth_to_pointcloud(depth_path, cam):
    img = cv2.imread(depth_path, cv2.IMREAD_COLOR)
    if img is None:
        return np.array([])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = decode_depth(img, cam["far_clip"])

    K = cam["K"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = depth.shape

    mask = (depth > cam["near_clip"] * 2) & (depth < cam["far_clip"] * 0.99)

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    pts_cam = np.stack([X, Y, Z], axis=-1)[mask]
    return pts_cam


def reconstruct_from_depth(folder_path):
    folder = Path(folder_path)
    cameras = load_camera_params(folder / "cameras.json")
    all_points = []

    for cam in cameras:
        depth_path = folder / cam["depth_name"]
        if not depth_path.exists():
            print(f"  [SKIP] depth manquante : {cam['depth_name']}")
            continue

        pts_cam = depth_to_pointcloud(str(depth_path), cam)
        if len(pts_cam) == 0:
            continue

        c2w = cam["c2w"]
        pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (c2w @ pts_h.T).T[:, :3]

        all_points.append(pts_world)
        print(f"  {cam['image_name']} → {len(pts_world)} points")
        break

    if not all_points:
        print(f"ERREUR : aucun point reconstruit pour {folder_path}")
        return None

    cloud = np.vstack(all_points)

    dist = np.linalg.norm(cloud, axis=1)
    mask = dist < np.percentile(dist, 98)
    cloud = cloud[mask]

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

    cloud = reconstruct_from_depth(folder_path)
    if cloud is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    save_ply(cloud, os.path.join(output_dir, f"{name}.ply"))
    save_off(cloud, os.path.join(output_dir, f"{name}.off"))
    print(f"  Sauvegardé → {output_dir}/{name}.off")