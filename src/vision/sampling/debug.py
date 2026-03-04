import cv2
import numpy as np
import json
import os
from pathlib import Path

def debug_transformation(folder_path, frame_i=0, frame_j=6):
    """
    Teste les 8 combinaisons de flip possibles sur 2 vues opposées.
    La bonne combinaison = les deux nuages se superposent.
    """
    folder = Path(folder_path)
    with open(folder / "cameras.json") as f:
        data = json.load(f)

    for flip_combo in range(8):
        fx_sign = 1 if not (flip_combo & 1) else -1
        fy_sign = 1 if not (flip_combo & 2) else -1
        fz_sign = 1 if not (flip_combo & 4) else -1

        all_pts = []
        for idx in [frame_i, frame_j]:
            cam_data = data[idx]
            w, h = cam_data["width"], cam_data["height"]
            fov = cam_data["fov"]
            fx = (w / 2) / np.tan(np.radians(fov / 2))
            K = np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1]], dtype=np.float64)

            vm = np.array(cam_data["view_matrix"]).reshape(4,4,order='F')
            c2w = np.linalg.inv(vm)

            # Applique le flip testé
            c2w[:, 0] *= fx_sign
            c2w[:, 1] *= fy_sign
            c2w[:, 2] *= fz_sign

            depth_path = folder / cam_data["depth_name"]
            img = cv2.imread(str(depth_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth = img[:,:,0].astype(np.float32) / 255.0 * cam_data["far_clip"]
            mask = (depth > 0.02) & (depth < cam_data["far_clip"] * 0.99)

            u, v = np.meshgrid(np.arange(w), np.arange(h))
            cx, cy = w/2, h/2
            X = (u - cx) * depth / fx
            Y = -(v - cy) * depth / fx
            Z = depth
            pts_cam = np.stack([X, Y, Z], axis=-1)[mask]

            pts_h = np.hstack([pts_cam, np.ones((len(pts_cam),1))])
            pts_world = (c2w @ pts_h.T).T[:, :3]
            all_pts.append(pts_world)

        # Distance entre les centres des deux nuages
        c1 = np.median(all_pts[0], axis=0)
        c2 = np.median(all_pts[1], axis=0)
        dist = np.linalg.norm(c1 - c2)
        print(f"Flip X={fx_sign:+d} Y={fy_sign:+d} Z={fz_sign:+d} → distance centres: {dist:.4f}")

if __name__ == "__main__":
    debug_transformation(r"C:\Users\fanny\OneDrive\Bureau\Cours_CS\DEEPL\ModelNet10-classifier\unity\Visual_V0\Assets\ModelsDatasetOutput\bathtub_0001")