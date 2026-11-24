#!/usr/bin/env python3
"""
train_robust.py: High-Accuracy, Fast & Robust Road Damage Detector.
Fine-tunes YOLO11s (official COCO weights) with domain-safe geometry
and 4-corruption robustness augmentations (blur, low-light, noise, JPEG).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# --- Hook Domain-Safe Robustness Augmentations into Ultralytics ---
import albumentations as A
from ultralytics.data import augment

_orig_alb_init = augment.Albumentations.__init__

def _custom_alb_init(self, p=1.0, transforms=None, flip_idx=None):
    if transforms is None:
        transforms = [
            A.OneOf([
                A.MotionBlur(blur_limit=(5, 11), p=1.0),
                A.GaussianBlur(blur_limit=(5, 11), p=1.0),
            ], p=0.20),
            A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.15), contrast_limit=(-0.25, 0.15), p=0.20),
            A.RandomGamma(gamma_limit=(60, 120), p=0.15),
            A.GaussNoise(std_range=(0.1, 0.25), p=0.15),
            A.ImageCompression(quality_range=(20, 50), p=0.20),
            A.CLAHE(clip_limit=(1.0, 3.0), p=0.10),
        ]
    return _orig_alb_init(self, p=p, transforms=transforms, flip_idx=flip_idx)

augment.Albumentations.__init__ = _custom_alb_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt", help="Official COCO pretrained weights")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workdir", default="robust_work")
    a = ap.parse_args()

    from ultralytics import YOLO

    work = Path(a.workdir).resolve()
    work.mkdir(exist_ok=True, parents=True)
    
    # Use the existing cached yolo_data
    data_yaml = Path("baseline_work/data.yaml").resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}. Run baseline data prep first.")

    print(f"Loading {a.model} (official COCO weights)...")
    model = YOLO(a.model)
    
    print("\nStarting Robust Road Damage Training...")
    print("Configuration:")
    print(f"  - Model: {a.model} (~9.4M params)")
    print(f"  - Epochs: {a.epochs}, Batch: {a.batch}, ImgSz: {a.imgsz}")
    print(f"  - Device: {a.device}, Workers: {a.workers}, Patience: {a.patience}")
    print("  - Domain-Safe: fliplr=0.5, flipud=0.0, degrees=0.0 (NO ROTATION)")
    print("  - Robustness Hook: MotionBlur, GaussianBlur, Brightness/Gamma, Noise, JPEG compression")
    print("  - Close Mosaic: last 10 epochs for precision box boundary learning")
    print("  - Cosine LR: cos_lr=True, max_det=100")

    model.train(
        data=str(data_yaml),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        patience=a.patience,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        # Domain-safe geometry: horizontal flip allowed, NEVER rotate or vertical flip
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        scale=0.5,
        translate=0.1,
        mosaic=1.0,
        mixup=0.1,
        max_det=100,
        project=str(work / "runs"),
        name="yolo11s_robust",
        exist_ok=True,
        plots=False,
    )

    out_weights = Path("weights")
    out_weights.mkdir(exist_ok=True)
    best_pt = model.trainer.best
    shutil.copy2(best_pt, out_weights / "best_yolo11s.pt")
    shutil.copy2(best_pt, out_weights / "best.pt")
    print(f"\nTraining Complete! Best model saved to: {out_weights / 'best_yolo11s.pt'} and {out_weights / 'best.pt'}")


if __name__ == "__main__":
    main()
