#!/usr/bin/env python3
"""
================================================================================
Anveshan DataSprint 2.0: Intelligent Road Damage Detection
Kaggle Cloud Training Pipeline: YOLO11m @ imgsz=736 (Multi-GPU T4 x2 / P100)
================================================================================
- Strictly 100% rules-compliant: Official COCO-pretrained weights only (yolo11m.pt)
- ZERO external road damage datasets
- Domain-safe geometry: degrees=0.0 (NO ROTATION), flipud=0.0, fliplr=0.5
- Optimized for Kaggle GPU T4 x2 (batch=32, DDP multi-GPU) or single GPU (batch=16)
- Automatically evaluates on val_images (FP16, imgsz=736) using official score.py
- Generates and verifies /kaggle/working/submission_yolo11m_736.csv
"""

import os
import sys
import subprocess
import shutil
import time
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 0. Environment Setup & Dependency Installation
# -----------------------------------------------------------------------------
print("=" * 80)
print("1. CHECKING ENVIRONMENT & INSTALLING DEPENDENCIES")
print("=" * 80)

try:
    import ultralytics
    print(f"Ultralytics already installed (v{ultralytics.__version__})")
except ImportError:
    print("Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    import ultralytics
    print(f"Installed Ultralytics v{ultralytics.__version__}")

from ultralytics import YOLO
import torch

# GPU Configuration
n_gpus = torch.cuda.device_count()
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Detected GPUs: {n_gpus}")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB)")

if n_gpus >= 2:
    device = list(range(n_gpus))
    batch_size = 32
    print(f"Multi-GPU Mode: Using device={device} with batch_size={batch_size}")
elif n_gpus == 1:
    device = 0
    batch_size = 16
    print(f"Single-GPU Mode: Using device={device} with batch_size={batch_size}")
else:
    device = "cpu"
    batch_size = 8
    print("WARNING: No GPU detected, running on CPU.")

IMGSZ = 736
EPOCHS = 40
MODEL_WEIGHTS = "yolo11m.pt"
WORKING_DIR = Path("/kaggle/working").resolve()
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Dataset Auto-Discovery under /kaggle/input
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. AUTO-DISCOVERING COMPETITION DATASET")
print("=" * 80)

def find_file(filename, search_root="/kaggle/input"):
    for root, _, files in os.walk(search_root):
        if filename in files:
            return Path(root) / filename
    return None

def find_dir(dirname, search_root="/kaggle/input"):
    for root, dirs, _ in os.walk(search_root):
        if dirname in dirs:
            return Path(root) / dirname
    return None

train_lbl_path = find_file("train_labels.csv")
val_lbl_path = find_file("val_labels.csv")
val_ids_path = find_file("val_image_ids.csv")
test_ids_path = find_file("test_image_ids.csv")
train_img_dir = find_dir("train_images")
val_img_dir = find_dir("val_images")
test_img_dir = find_dir("test_images")

if not train_lbl_path or not train_img_dir:
    # Local fallback for testing outside kaggle
    print("Warning: /kaggle/input not found, falling back to local current directory...")
    train_lbl_path = Path("train_labels.csv").resolve()
    val_lbl_path = Path("val_labels.csv").resolve()
    val_ids_path = Path("val_image_ids.csv").resolve()
    test_ids_path = Path("test_image_ids.csv").resolve()
    train_img_dir = Path("train_images").resolve()
    val_img_dir = Path("val_images").resolve()
    test_img_dir = Path("test_images").resolve()
    WORKING_DIR = Path("kaggle_working").resolve()
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

print(f"Found train_labels.csv: {train_lbl_path}")
print(f"Found val_labels.csv:   {val_lbl_path}")
print(f"Found val_image_ids.csv:{val_ids_path}")
print(f"Found test_image_ids.csv:{test_ids_path}")
print(f"Found train_images/:    {train_img_dir}")
print(f"Found val_images/:      {val_img_dir}")
print(f"Found test_images/:     {test_img_dir}")

# -----------------------------------------------------------------------------
# 2. Building YOLO Dataset (Symlinks + Normalized Labels)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. BUILDING YOLO DATASET WITH BACKGROUND NEGATIVES")
print("=" * 80)

dataset_dir = WORKING_DIR / "yolo_dataset"

def prepare_yolo_split(img_dir: Path, lbl_csv: Path, split_name: str, out_root: Path):
    images_dst = out_root / "images" / split_name
    labels_dst = out_root / "labels" / split_name
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    boxes = defaultdict(list)
    if lbl_csv and lbl_csv.exists():
        with open(lbl_csv, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                boxes[r["image_id"]].append((
                    int(r["class_id"]),
                    float(r["x_min"]),
                    float(r["y_min"]),
                    float(r["x_max"]),
                    float(r["y_max"]),
                ))

    all_images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    total_imgs = len(all_images)
    empty_imgs = 0

    print(f"Processing split '{split_name}': {total_imgs} images...")

    for p in all_images:
        link_target = images_dst / p.name
        if not link_target.exists():
            try:
                os.symlink(p.resolve(), link_target)
            except OSError:
                try:
                    os.link(p.resolve(), link_target)
                except OSError:
                    shutil.copy2(p, link_target)

        W, H = Image.open(p).size
        lines = []
        img_boxes = boxes.get(p.stem, [])
        if not img_boxes:
            empty_imgs += 1

        for c, x1, y1, x2, y2 in img_boxes:
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        lbl_file = labels_dst / f"{p.stem}.txt"
        lbl_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding='utf-8')

    print(f"Split '{split_name}' complete: {total_imgs} images ({empty_imgs} background empty images preserved)")
    return total_imgs

n_train = prepare_yolo_split(train_img_dir, train_lbl_path, "train", dataset_dir)
n_val = prepare_yolo_split(val_img_dir, val_lbl_path, "val", dataset_dir)

# Create data.yaml
data_yaml_path = dataset_dir / "data.yaml"
data_yaml_content = f"""path: {dataset_dir.as_posix()}
train: images/train
val: images/val
names:
  0: longitudinal_crack
  1: transverse_crack
  2: alligator_crack
  3: pothole
"""
data_yaml_path.write_text(data_yaml_content, encoding='utf-8')
print(f"\nWrote data.yaml to {data_yaml_path}")

# -----------------------------------------------------------------------------
# 3. Model Training (YOLO11m @ imgsz=736)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"4. TRAINING YOLO11m (imgsz={IMGSZ}, epochs={EPOCHS}, batch={batch_size})")
print("=" * 80)

model = YOLO(MODEL_WEIGHTS)

runs_project = WORKING_DIR / "runs"
train_name = "yolo11m_736"

results = model.train(
    data=str(data_yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=batch_size,
    device=device,
    amp=True,
    cos_lr=True,
    close_mosaic=10,
    patience=12,
    # Domain-safe geometry (CRITICAL: ZERO ROTATION to preserve D00 longitudinal vs D10 transverse)
    degrees=0.0,
    flipud=0.0,
    fliplr=0.5,
    scale=0.5,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    mosaic=1.0,
    mixup=0.1,
    max_det=100,
    project=str(runs_project),
    name=train_name,
    exist_ok=True,
    plots=False,
)

# Preserve best checkpoint
best_pt = model.trainer.best
final_best_pt = WORKING_DIR / "best_yolo11m_736.pt"
shutil.copy2(best_pt, final_best_pt)
print(f"\n[OK] Training completed! Best model saved to: {final_best_pt}")

# -----------------------------------------------------------------------------
# 4. Evaluation on val_images & Local CV Scoring
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("5. EVALUATING ON val_images (FP16, imgsz=736, conf=0.001, iou=0.60)")
print("=" * 80)

eval_model = YOLO(str(final_best_pt))
try:
    eval_model.fuse()
except Exception:
    pass

val_ids = []
with open(val_ids_path, newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        val_ids.append(r["image_id"])

val_pred_rows = []
eval_device = 0 if torch.cuda.is_available() else "cpu"

print(f"Predicting on {len(val_ids)} validation images...")
t_val_start = time.time()

for idx, iid in enumerate(val_ids):
    img_p = val_img_dir / f"{iid}.jpg"
    if not img_p.exists():
        img_p = val_img_dir / f"{iid}.png"

    with torch.inference_mode():
        res = eval_model.predict(
            source=str(img_p),
            imgsz=IMGSZ,
            conf=0.001,
            iou=0.60,
            max_det=100,
            device=eval_device,
            half=(eval_device != "cpu"),
            verbose=False,
        )[0]

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls_arr = res.boxes.cls.cpu().numpy().astype(int)

        order = np.argsort(-conf)[:100]
        for oi in order:
            val_pred_rows.append([
                iid,
                int(cls_arr[oi]),
                f"{float(conf[oi]):.5f}",
                f"{float(xyxy[oi][0]):.2f}",
                f"{float(xyxy[oi][1]):.2f}",
                f"{float(xyxy[oi][2]):.2f}",
                f"{float(xyxy[oi][3]):.2f}",
            ])

val_pred_csv = WORKING_DIR / "val_pred_yolo11m_736.csv"
with open(val_pred_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"])
    writer.writerows(val_pred_rows)

print(f"Validation predictions written: {len(val_pred_rows)} boxes in {time.time()-t_val_start:.1f}s")

# Run score.py if found, or inline scorer
score_script = find_file("score.py")
if score_script and score_script.exists():
    print(f"\nRunning official scorer: {score_script}")
    subprocess.run([
        sys.executable, str(score_script),
        "--pred", str(val_pred_csv),
        "--gt", str(val_lbl_path),
        "--image-ids", str(val_ids_path)
    ])

# -----------------------------------------------------------------------------
# 5. Full Test Inference & Kaggle Submission Generation
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("6. RUNNING FP16 TEST INFERENCE & GENERATING SUBMISSION")
print("=" * 80)

test_ids = []
with open(test_ids_path, newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        test_ids.append(r["image_id"])

total_test = len(test_ids)
print(f"Predicting on {total_test} test images...")

test_pred_boxes = defaultdict(list)
t_test_start = time.time()

batch_eval_size = 16
for b_idx in range(0, total_test, batch_eval_size):
    b_ids = test_ids[b_idx : b_idx + batch_eval_size]
    b_paths = []
    for iid in b_ids:
        p = test_img_dir / f"{iid}.jpg"
        if not p.exists():
            p = test_img_dir / f"{iid}.png"
        b_paths.append(str(p))

    with torch.inference_mode():
        results = eval_model.predict(
            source=b_paths,
            imgsz=IMGSZ,
            conf=0.001,
            iou=0.60,
            max_det=100,
            device=eval_device,
            half=(eval_device != "cpu"),
            verbose=False,
            batch=len(b_paths),
        )

    for iid, r in zip(b_ids, results):
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)

            order = np.argsort(-conf)[:100]
            for oi in order:
                test_pred_boxes[iid].append((
                    int(cls_arr[oi]),
                    float(conf[oi]),
                    float(xyxy[oi][0]),
                    float(xyxy[oi][1]),
                    float(xyxy[oi][2]),
                    float(xyxy[oi][3]),
                ))

    done = min(b_idx + batch_eval_size, total_test)
    if done % 500 == 0 or done == total_test:
        elapsed = time.time() - t_test_start
        print(f"Test progress: {done}/{total_test} images ({done/total_test*100:.1f}%) in {elapsed:.1f}s")

# Format into Kaggle Submission CSV
submission_csv = WORKING_DIR / "submission_yolo11m_736.csv"
sub_rows = []
total_detections = 0
empty_test_images = 0

for iid in test_ids:
    dets = test_pred_boxes.get(iid, [])
    if not dets:
        sub_rows.append([iid, "none"])
        empty_test_images += 1
    else:
        # Format: class score x_min y_min x_max y_max ...
        tokens = []
        for c, s, x1, y1, x2, y2 in dets:
            tokens.extend([str(c), f"{s:.5f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])
            total_detections += 1
        sub_rows.append([iid, " ".join(tokens)])

with open(submission_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "PredictionString"])
    writer.writerows(sub_rows)

print(f"\n[OK] Submission generated: {submission_csv}")
print(f"Total detections: {total_detections} across {total_test - empty_test_images} images ({empty_test_images} empty)")

# Validate with check_submission.py
check_script = find_file("check_submission.py")
if check_script and check_script.exists():
    print(f"\nRunning submission validator: {check_script}")
    subprocess.run([
        sys.executable, str(check_script),
        "--pred", str(submission_csv),
        "--image-ids", str(test_ids_path)
    ])

print("\n" + "=" * 80)
print("TRAINING, EVALUATION, AND SUBMISSION GENERATION COMPLETE!")
print("Artifacts generated:")
print(f"  - Model:      {final_best_pt}")
print(f"  - Submission: {submission_csv}")
print("=" * 80)
