"""Road Damage Detector module for civic reporting app.

Runs inference strictly on CPU using PyTorch and Ultralytics YOLO.
Automatically selects best_yolo11s.pt if available, falling back to baseline weights/best.pt.
Draws color-coded bounding boxes and computes repair urgency priority.
"""

from io import BytesIO
import base64
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from ultralytics import YOLO

CLASS_METADATA = {
    0: {
        "code": "D00",
        "name": "Longitudinal Crack",
        "description": "Crack parallel to the roadway centerline, typically caused by pavement fatigue, joint construction, or shrinkage.",
        "color": (59, 130, 246),       # Blue #3B82F6
        "text_color": (255, 255, 255),
    },
    1: {
        "code": "D10",
        "name": "Transverse Crack",
        "description": "Crack perpendicular to roadway centerline, typically caused by thermal contraction or base layer reflection.",
        "color": (16, 185, 129),       # Emerald Green #10B981
        "text_color": (255, 255, 255),
    },
    2: {
        "code": "D20",
        "name": "Alligator Crack",
        "description": "Interconnected fatigue cracking resembling reptile scales, indicating structural structural sub-base failure.",
        "color": (245, 158, 11),       # Amber / Orange #F59E0B
        "text_color": (0, 0, 0),
    },
    3: {
        "code": "D40",
        "name": "Pothole",
        "description": "Severe bowl-shaped cavity penetrating the asphalt surface, creating immediate vehicular hazard and tire damage risk.",
        "color": (239, 68, 68),        # Red #EF4444
        "text_color": (255, 255, 255),
    },
}

DEFAULT_CANDIDATE_WEIGHTS = [
    os.path.join(os.path.dirname(__file__), "..", "weights", "best_yolo11s.pt"),
    os.path.join(os.path.dirname(__file__), "..", "weights", "best.pt"),
    os.path.join(os.path.dirname(__file__), "..", "runs", "train_baseline", "weights", "best.pt"),
]


class RoadDamageDetector:
    """CPU-safe Road Damage Detector for Civic Grievance Reporting."""

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = self._resolve_weights_path(weights_path)
        self.model = None
        self._load_model()

    def _resolve_weights_path(self, explicit_path: Optional[str]) -> str:
        if explicit_path and os.path.exists(explicit_path):
            return os.path.abspath(explicit_path)

        for candidate in DEFAULT_CANDIDATE_WEIGHTS:
            norm_path = os.path.abspath(candidate)
            if os.path.exists(norm_path):
                return norm_path

        raise FileNotFoundError(
            f"No trained YOLO weights found. Looked in: {[os.path.abspath(c) for c in DEFAULT_CANDIDATE_WEIGHTS]}"
        )

    def _load_model(self):
        """Loads YOLO model strictly on CPU."""
        print(f"[CivicDetector] Loading weights from: {self.weights_path} on CPU")
        # Ensure device="cpu" is enforced
        self.model = YOLO(self.weights_path)
        try:
            self.model.to("cpu")
        except Exception:
            pass

    def reload_if_better_weights_available(self):
        """Checks if best_yolo11s.pt has completed and updates model path if currently on baseline."""
        yolo11s_path = os.path.abspath(DEFAULT_CANDIDATE_WEIGHTS[0])
        if os.path.exists(yolo11s_path) and self.weights_path != yolo11s_path:
            print(f"[CivicDetector] Upgrading active model to completed YOLO11s weights: {yolo11s_path}")
            self.weights_path = yolo11s_path
            self._load_model()

    def detect_and_annotate(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Runs road distress detection on raw image bytes.

        Returns annotated image (base64 JPEG), damage breakdown, and priority score.
        """
        self.reload_if_better_weights_available()

        # Load and orient image properly using PIL EXIF orientation
        raw_pil = Image.open(BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(raw_pil).convert("RGB")
        width, height = pil_img.size

        # Run inference strictly in inference mode on CPU
        with torch.inference_mode():
            results = self.model.predict(
                source=pil_img,
                device="cpu",
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=640,
                verbose=False,
            )

        boxes = results[0].boxes
        detections: List[Dict[str, Any]] = []

        counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}

        draw_img = pil_img.copy()
        draw = ImageDraw.Draw(draw_img)

        # Dynamic box thickness based on image resolution
        box_thickness = max(3, int(min(width, height) / 200))

        # Attempt to load default font with scalable size
        font_size = max(14, int(min(width, height) / 45))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        if boxes is not None and len(boxes) > 0:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy_arr)):
                cls_id = int(cls_arr[i])
                confidence = float(conf_arr[i])
                x1, y1, x2, y2 = [float(v) for v in xyxy_arr[i]]

                meta = CLASS_METADATA.get(cls_id, {
                    "code": f"C{cls_id}",
                    "name": "Surface Distress",
                    "description": "General road surface irregularity",
                    "color": (255, 255, 0),
                    "text_color": (0, 0, 0),
                })

                code = meta["code"]
                if code in counts:
                    counts[code] += 1

                detections.append({
                    "class_id": cls_id,
                    "code": code,
                    "name": meta["name"],
                    "confidence": round(confidence, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "description": meta["description"],
                })

                # Draw bounding box
                color = meta["color"]
                for t in range(box_thickness):
                    draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

                # Label text and background tag
                label_text = f"{code} {meta['name']} {int(confidence * 100)}%"
                bbox_text = draw.textbbox((x1, y1), label_text, font=font)
                text_w = bbox_text[2] - bbox_text[0]
                text_h = bbox_text[3] - bbox_text[1]

                tag_y1 = max(0.0, y1 - text_h - 8)
                tag_y2 = tag_y1 + text_h + 8
                tag_x2 = min(float(width), x1 + text_w + 12)

                draw.rectangle([x1, tag_y1, tag_x2, tag_y2], fill=color)
                draw.text(
                    (x1 + 6, tag_y1 + 4),
                    label_text,
                    fill=meta["text_color"],
                    font=font,
                )

        # Compute Urgency Priority
        total_damage = len(detections)
        if total_damage == 0:
            priority = "None"
            priority_color = "#10B981"
            urgency_note = "No significant structural pavement distress detected above confidence threshold."
        elif counts["D40"] > 0 or total_damage >= 3:
            priority = "High"
            priority_color = "#EF4444"
            if counts["D40"] > 0:
                urgency_note = f"CRITICAL: {counts['D40']} pothole(s) detected. High risk of vehicular accident, vehicle axle damage, or citizen injury."
            else:
                urgency_note = f"HIGH: Multiple distress areas ({total_damage} damage zones) requiring immediate municipal intervention before road foundation collapses."
        elif counts["D20"] > 0:
            priority = "Medium"
            priority_color = "#F59E0B"
            urgency_note = f"MODERATE: Structural fatigue ({counts['D20']} alligator cracking area) indicates base-course water percolation and impending pothole formation."
        else:
            priority = "Low"
            priority_color = "#3B82F6"
            urgency_note = f"ROUTINE: Surface crack distress ({counts['D00']} longitudinal, {counts['D10']} transverse). Sealant crack-filling recommended during scheduled preventive maintenance."

        # Encode annotated image to Base64 JPEG
        out_buf = BytesIO()
        draw_img.save(out_buf, format="JPEG", quality=88)
        b64_str = base64.b64encode(out_buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_str}"

        # Human-readable summary
        summary_parts = []
        if counts["D40"] > 0:
            summary_parts.append(f"{counts['D40']} Pothole(s)")
        if counts["D20"] > 0:
            summary_parts.append(f"{counts['D20']} Alligator Crack(s)")
        if counts["D00"] > 0:
            summary_parts.append(f"{counts['D00']} Longitudinal Crack(s)")
        if counts["D10"] > 0:
            summary_parts.append(f"{counts['D10']} Transverse Crack(s)")

        if not summary_parts:
            damage_summary = "Clear Road Surface (0 defects)"
        else:
            damage_summary = ", ".join(summary_parts)

        return {
            "model_version": os.path.basename(self.weights_path),
            "image_dimensions": {"width": width, "height": height},
            "annotated_image_data_url": data_url,
            "annotated_bytes": out_buf.getvalue(),
            "detections": detections,
            "counts": counts,
            "total_damage_count": total_damage,
            "priority": priority,
            "priority_color": priority_color,
            "urgency_note": urgency_note,
            "damage_summary": damage_summary,
        }


# Lazy singleton instance
_DETECTOR_INSTANCE: Optional[RoadDamageDetector] = None


def get_detector() -> RoadDamageDetector:
    global _DETECTOR_INSTANCE
    if _DETECTOR_INSTANCE is None:
        _DETECTOR_INSTANCE = RoadDamageDetector()
    return _DETECTOR_INSTANCE
