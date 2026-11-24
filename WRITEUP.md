# Anveshan DataSprint 2.0: Intelligent Road Damage Detection
## Technical Write-Up & Methodology Report

---

### 1. Executive Summary & Problem Framing
The **Anveshan DataSprint 2.0** challenge requires detecting, localizing, and classifying four critical categories of pavement surface distress from roadway photography under real-world weather, lighting, and sensor degradation:
* **D00:** Longitudinal Cracking (cracks parallel to roadway direction)
* **D10:** Transverse Cracking (cracks perpendicular to roadway direction)
* **D20:** Alligator Fatigue Cracking (interconnected sub-base failure)
* **D40:** Potholes (deep asphalt cavities posing vehicle safety hazard)

#### Competition Evaluation Metric (100-Point Composite Score)
The competition ranks submissions based on a composite metric designed to reward holistic engineering:
$$\text{Composite Score} = 0.30 \times \text{mAP@50} + 0.25 \times \text{mAP@50:95} + 0.15 \times \text{Precision} + 0.15 \times \text{Recall} + 0.10 \times \text{Speed Score} + 0.05 \times \text{Robustness mAP@50}$$
Where:
* **Speed Score:** Evaluated at batch size 1 on an NVIDIA T4 GPU ($\le 25\text{ ms} \rightarrow 1.0\text{ score}$, $\ge 250\text{ ms} \rightarrow 0.0\text{ score}$).
* **Robustness Score:** Evaluated on degraded test images exhibiting motion/optical blur, low light/underexposure, sensor noise, and heavy JPEG compression artifacts.
* **Format:** Every test image is predicted with bounding boxes down to confidence `0.001`, or the literal word `none` for undamaged roads.

---

### 2. Strict Compliance & Pretrained Weights Disclosure
* **Zero External Road Datasets:** No external road, pavement, asphalt, or crack imagery/labels were used during any stage of training. Training was conducted exclusively on the official competition split (`14,691` train images).
* **Approved Pretrained Backbones Only:** In accordance with competition rules, only official generic backbones pretrained strictly on COCO (common objects) were used:
  1. `yolov8n.pt` (Ultralytics YOLOv8 Nano, COCO pretrained, 3.2M parameters) — used for Part 1 baseline.
  2. `yolo11s.pt` (Ultralytics YOLO11 Small, COCO pretrained, 9.4M parameters, 21.7 GFLOPs) — used for Part 2 robust model.
* **No Domain Pretrained Weights:** No RDD2020, CRDDC, or pavement-specific pretrained checkpoints were used.

---

### 3. Data Audit & Key Structural Findings
A comprehensive pre-training audit of `train_labels.csv` (`14,691` images) and `val_labels.csv` (`2,150` images) revealed:
1. **High Negative (Undamaged) Ratio:** Exactly **42.0%** of train images (6,168 images) and **41.7%** of val images (896 images) contain zero damage annotations. Retaining all background images during training was essential to prevent false-positive hallucinations on undamaged tarmac.
2. **Aspect Ratio Consistency:** 100% of competition photos are perfectly square (`600x600` or `720x720`). An input resolution of `imgsz=640` aligns with native resolution while maximizing inference speed.
3. **Class Imbalance:** Potholes (`D40`) and alligator cracking (`D20`) are sparser and geometrically localized compared to widespread longitudinal (`D00`) and transverse (`D10`) cracks.

---

### 4. Part 1 Baseline Results (Public Leaderboard Rank #9)
The baseline pipeline was trained with `yolov8n.pt` for 10 epochs using standard hyperparameters:
* **Local Validation Score (`score.py`):**
  * $\text{mAP@50} = \mathbf{0.4007}$ | $\text{mAP@50:95} = \mathbf{0.1657}$ | $\text{Precision} = \mathbf{0.4436}$ | $\text{Recall} = \mathbf{0.4312}$
  * **Composite Accuracy Score: 34.45 / 100**
  * **Baseline Speed:** $35.83\text{ ms/image}$ on RTX 3050 Laptop ($\text{Speed Score} = 0.8437$)
* **Kaggle Public Leaderboard Submission:**
  * **Public Leaderboard Score: 34.36786**
  * **Current Standing: Rank #9 on Kaggle Leaderboard**
  * Verified $0.08\%$ consistency between local validation CV (34.45) and Kaggle Public LB (34.37).
* **Baseline Robustness Vulnerability:**
  * When evaluated on a deterministic degraded validation benchmark (`val_images_degraded/`), the baseline mAP@50 dropped sharply from **0.4007** down to **0.1537** (a $61.6\%$ drop), highlighting the urgent need for domain-safe robustness training.

---

### 5. Part 2: Architecture & Domain-Safe Robustness Engineering

#### 5.1 Architecture Selection: YOLO11 Small (`yolo11s`)
We selected `yolo11s` for the competitive pipeline:
* **Parameters & GFLOPs:** 9.4M parameters, 21.7 GFLOPs.
* **Structural Upgrades:** Utilizes improved C3k2 cross-stage partial bottlenecks, Spatial Pyramid Pooling Fast (SPPF), and an anchor-free decoupled head for tighter bounding box regression, specifically targeting the $25\%$ weight on mAP@50:95.

#### 5.2 Critical Domain Geometry Rule: Zero Rotation
* In standard object detection, rotational augmentations (`degrees > 0`) are ubiquitous.
* **In Road Damage Detection, rotation is fatal:** Rotating an image by $90^\circ$ converts longitudinal cracks (`D00`) into transverse cracks (`D10`), and vice-versa, fundamentally corrupting feature representations.
* **Strict Constraint:** We enforced `degrees = 0.0`, `flipud = 0.0`, and allowed only horizontal reflection (`fliplr = 0.5`), scale (`scale = 0.5`), and translation (`translate = 0.1`).

#### 5.3 4-Corruption Robustness Augmentation Pipeline
To directly optimize the competition's 5% Robustness score and prevent real-world performance degradation, we integrated an online custom augmentation pipeline injecting the four competition degradations:
1. **Motion & Gaussian Blur ($p=0.20$):** Simulates vehicle movement vibration and lens defocus.
2. **Low-Light & Underexposure ($p=0.20$):** Random gamma reduction and contrast drop simulating dusk, shadows, and overcast sky.
3. **Sensor Noise ($p=0.20$):** Gaussian and ISO camera sensor noise.
4. **Heavy JPEG Compression ($p=0.20$):** Quality range $15–40$ simulating aggressive compression artifacts.
* **Uncorrupted Anchor Policy:** $>30\%$ of training samples remain uncorrupted so clean daylight performance is not compromised.

#### 5.4 Training Recipe & Convergence Control
* **Optimizer:** AdamW with cosine learning rate scheduling (`cos_lr=True`).
* **Mosaic Schedule:** `mosaic=1.0` during early feature formation, strictly turned off during the final 10 epochs (`close_mosaic=10`) to allow fine-grained localization of crack boundaries.

---

### 6. Inference Optimization & 100% Offline Contract (`inference.py`)
To guarantee reproducible, deterministic execution under the competition's offline T4 evaluation:
1. **Zero External Dependencies:** Relies strictly on standard PyTorch and Ultralytics (no brittle external custom plugins).
2. **Conv + BatchNorm Fusion:** `model.fuse()` eliminates batch normalization overhead during evaluation.
3. **Half-Precision (FP16):** Tensor cores utilized via `half=True` on CUDA, cutting latency while preserving box accuracy.
4. **Warmup Passes:** 3 dummy forward passes inside `load(device)` eliminate PyTorch CUDA runtime initialization latency from the timed benchmark loop.
5. **Prediction Post-Processing:** Run under `torch.inference_mode()` with `conf=0.001`, `max_det=100`, and tuned NMS `iou=0.60`.

---

### 7. Results & Comparative Benchmarks

| Metric | Part 1 Baseline (`yolov8n.pt`) | Part 2 Final Model (`yolo11s.pt`, Ep 21) | Improvement |
| :--- | :---: | :---: | :---: |
| **mAP@50 (Clean Val)** | 0.4007 | **0.4408** | **+0.0401 (+10.0%)** |
| **mAP@50:95 (Clean Val)** | 0.1657 | **0.1881** | **+0.0224 (+13.5%)** |
| **Precision** | 0.4436 | **0.5146** | **+0.0710 (+16.0%)** |
| **Recall** | 0.4312 | **0.4259** | -0.0053 |
| **Accuracy Score (`score.py` / 100)** | **34.45 / 100** | **37.68 / 100** | **+3.23 points** |
| **Robustness mAP@50 (Degraded Val)**| 0.1537 | **0.2243** | **+0.0706 (+45.9% relative)** |
| **Median Latency (RTX 3050, bs=1)**| 35.83 ms (`0.8437`) | **38.87 ms (`0.8083`)** | FP16 + fused Conv+BN |
| **Full 100-Pt Composite (RTX 3050)**| 38.49 / 100 | **41.24 / 100** | **+2.75 points (+43.16 est. T4)** |
| **Kaggle Public Leaderboard** | **34.36786** | **36.52343 (Ep 19) / 36.38536 (Ep 21)** | **+2.1556 LB jump (Rank #15)** |

---

### 8. System Environment & Reproducibility Specifications
* **Operating System:** Windows 11 Home (Build 26100)
* **Compute Device:** NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MiB GDDR6 VRAM)
* **CUDA / Driver:** CUDA 12.4 / Driver Version 610.91
* **Python Runtime:** Python 3.12 (64-bit)
* **Core Libraries:**
  * `torch == 2.6.0+cu124`
  * `torchvision == 0.21.0+cu124`
  * `ultralytics == 8.4.144`
  * `albumentations == 2.0.8`
  * `pillow == 13.0.0`
  * `pandas == 2.3.3`
  * `numpy == 2.4.6`

---

### 9. Beyond Competition: Citizen-to-Municipal Action Portal (`app/`)
To demonstrate practical civic utility beyond leaderboard scoring, the trained model was integrated into a standalone civic grievance web application (`app/`):
* **Decoupled Architecture:** Runs 100% independently on CPU, ensuring zero interference with root competition scripts.
* **Citizen Mobile Workflow:** One-tap camera capture + browser Geolocation API (`navigator.geolocation`) generates GPS coordinates and Google Maps pin links.
* **Automated Municipal Routing:** Reverse-geocodes locations via OpenStreetMap Nominatim and matches incidents to official grievance cells (e.g., BBMP Bengaluru, MCC Mysuru, MCD Delhi, BMC Mumbai).
* **Automated Formal Dispatch:** Generates formal complaint drafts and dispatches them with attached AI-annotated photographic evidence via Gmail SMTP (TLS).
