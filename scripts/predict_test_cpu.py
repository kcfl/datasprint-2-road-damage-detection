import os
import sys
import csv
import time
import numpy as np
from PIL import Image

# Force complete isolation from GPU to guarantee zero CUDA memory impact
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import inference

def main():
    print("=== Running CPU-Only Inference on test_images/ (Baseline YOLOv8n) ===")
    img_dir = "test_images"
    test_ids_file = "test_image_ids.csv"
    out_pred_csv = "test_pred_baseline.csv"
    
    # Load all test IDs
    with open(test_ids_file, newline='') as f:
        reader = csv.DictReader(f)
        test_ids = [r["image_id"] for r in reader]
    total_imgs = len(test_ids)
    print(f"Total test images to predict: {total_imgs}")
    
    # Load model strictly on CPU
    t0 = time.time()
    model = inference.load("cpu")
    print(f"Model loaded on CPU in {time.time()-t0:.2f}s")
    
    pred_rows = []
    total_boxes = 0
    t_start = time.time()
    
    for idx, iid in enumerate(test_ids):
        img_path = os.path.join(img_dir, f"{iid}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{iid}.png")
            
        img = np.asarray(Image.open(img_path).convert("RGB"))
        dets = inference.predict(model, img)
        
        # Keep top 100 detections
        dets = sorted(dets, key=lambda x: -x[1])[:100]
        
        for c, s, x1, y1, x2, y2 in dets:
            pred_rows.append([iid, int(c), f"{s:.5f}", f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"])
            total_boxes += 1
            
        if (idx + 1) % 500 == 0 or (idx + 1) == total_imgs:
            elapsed = time.time() - t_start
            ips = (idx + 1) / elapsed
            eta_sec = (total_imgs - (idx + 1)) / max(ips, 1e-5)
            print(f"Progress: {idx+1}/{total_imgs} images ({((idx+1)/total_imgs)*100:.1f}%) - Speed: {ips:.1f} img/s - Total Boxes: {total_boxes} - ETA: {eta_sec:.1f}s")
            
    # Write intermediate detections CSV
    with open(out_pred_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"])
        writer.writerows(pred_rows)
        
    print(f"\nWrote raw predictions to {out_pred_csv}: {total_boxes} boxes across {total_imgs} images in {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()
