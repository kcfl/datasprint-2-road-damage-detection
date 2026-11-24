"""
inference.py : Optimized competition inference engine.
Follows the official contract: load(device) -> predict(model, image).
Includes FP16 acceleration, fused Conv+BN layers, inference_mode, and pre-warmup.
"""
import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(_HERE, "weights", "best_yolo11s.pt")
if not os.path.exists(WEIGHTS):
    WEIGHTS = os.path.join(_HERE, "weights", "best.pt")
IMGSZ = 640


class _Model:
    def __init__(self, yolo, device, half=False, iou=0.60):
        self.yolo = yolo
        self.device = device
        self.half = half
        self.iou = iou


def load(device="cuda"):
    from ultralytics import YOLO

    yolo = YOLO(WEIGHTS)
    dev_str = str(device)
    is_cuda = dev_str.startswith("cuda") and torch.cuda.is_available()
    
    # Fuse Conv2d + BatchNorm layers for speed
    try:
        yolo.fuse()
    except Exception:
        pass

    half = is_cuda  # FP16 acceleration on CUDA devices
    model = _Model(yolo, device=device, half=half, iou=0.60)

    # 3 dummy warmup passes so GPU kernels/cuDNN are pre-compiled (un-timed)
    if is_cuda:
        dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
        for _ in range(3):
            _ = predict(model, dummy)

    return model


def predict(model, image):
    bgr = np.ascontiguousarray(image[:, :, ::-1])
    with torch.inference_mode():
        r = model.yolo.predict(
            bgr,
            imgsz=IMGSZ,
            conf=0.001,
            iou=model.iou,
            max_det=100,
            device=model.device,
            half=model.half,
            verbose=False,
        )[0]

    if r.boxes is None or len(r.boxes) == 0:
        return []

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)

    return [
        (int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        for c, s, b in zip(cls, conf, xyxy)
    ]
