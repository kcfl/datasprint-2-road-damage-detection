"""CPU-Only Test Inference for YOLO11s Mid-Training Snapshot.

Completely isolated from CUDA to ensure ZERO interference with the running
GPU training task (task-1120).
"""

import os
import sys
# Force complete isolation from GPU to guarantee zero CUDA memory impact
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
import time
import numpy as np
from PIL import Image

def main():
    print("=== Running CPU-Only Inference on test_images/ (YOLO11s Mid Snapshot) ===")
    img_dir = "test_images"
    test_ids_file = "test_image_ids.csv"
    weights_path = os.path.join("weights", "yolo11s_mid_snapshot.pt")
    out_pred_csv = "test_pred_yolo11s_mid.csv"
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")
        
    # Load all test IDs
    with open(test_ids_file, newline='') as f:
        reader = csv.DictReader(f)
        test_ids = [r["image_id"] for r in reader]
    total_imgs = len(test_ids)
    print(f"Total test images to predict: {total_imgs}")
    
    from ultralytics import YOLO
    import torch
    
    t0 = time.time()
    model = YOLO(weights_path)
    try:
        model.fuse()
    except Exception:
        pass
    print(f"Model loaded on CPU in {time.time()-t0:.2f}s")
    
    pred_rows = []
    total_boxes = 0
    t_start = time.time()
    
    # Process images with progress reporting
    batch_size = 16
    for b_idx in range(0, total_imgs, batch_size):
        b_ids = test_ids[b_idx : b_idx + batch_size]
        b_paths = []
        for iid in b_ids:
            p = os.path.join(img_dir, f"{iid}.jpg")
            if not os.path.exists(p):
                p = os.path.join(img_dir, f"{iid}.png")
            b_paths.append(p)
            
        with torch.inference_mode():
            results = model.predict(
                source=b_paths,
                imgsz=640,
                conf=0.001,
                iou=0.60,
                max_det=100,
                device="cpu",
                half=False,
                verbose=False,
                batch=len(b_paths),
            )
            
        for iid, r in zip(b_ids, results):
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                
                # Keep top 100 detections sorted by confidence
                order = np.argsort(-conf)[:100]
                for idx in order:
                    c = int(cls[idx])
                    s = float(conf[idx])
                    x1, y1, x2, y2 = xyxy[idx]
                    pred_rows.append([iid, c, f"{s:.5f}", f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"])
                    total_boxes += 1

        done = min(b_idx + batch_size, total_imgs)
        if done % 200 == 0 or done == total_imgs or done == batch_size:
            elapsed = time.time() - t_start
            ips = done / max(elapsed, 1e-5)
            eta_sec = (total_imgs - done) / max(ips, 1e-5)
            print(f"Progress: {done}/{total_imgs} images ({done/total_imgs*100:.1f}%) - Speed: {ips:.1f} img/s - Total Boxes: {total_boxes} - ETA: {eta_sec/60:.1f} min ({eta_sec:.0f}s)")

    # Write intermediate detections CSV
    with open(out_pred_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"])
        writer.writerows(pred_rows)
        
    print(f"\nWrote raw predictions to {out_pred_csv}: {total_boxes} boxes across {total_imgs} images in {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()
