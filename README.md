# DataSprint 2.0: Intelligent Road Damage Detection, starter kit

Detect and classify road damage in photos: longitudinal cracks (D00), transverse cracks (D10), alligator cracks (D20) and potholes (D40).
The full rules, metrics and scoring are on the competition page. This file lists what is in the kit and how to run it.

## What is in the kit

| File or folder | Contents |
| --- | --- |
| `train_images/`, `train_labels.csv` | 14691 images with labels. One row per damage box: `image_id,class_id,x_min,y_min,x_max,y_max`. Photos with no damage have no rows |
| `val_images/`, `val_labels.csv`, `val_image_ids.csv` | 2150 images with labels, for choosing models and for the local scorer |
| `test_images/`, `test_image_ids.csv` | 4200 images, inputs only (normal and degraded photos mixed). You predict boxes for every one |
| `score.py`, `check_submission.py` | The scorer and the format check. The leaderboard uses the same scoring code |
| `inference.py`, `benchmark_speed.py` | The baseline model code with the rules it must follow, and the speed benchmark |
| `train_baseline.py`, `make_kaggle_submission.py` | Trains the baseline, and turns a one-row-per-box file into the Kaggle upload |
| `requirements.txt` | Library versions used to build and test the kit |

Class ids: 0 = D00 longitudinal crack, 1 = D10 transverse crack, 2 = D20 alligator crack, 3 = D40 pothole.
If the data arrives as a zip, unzip it first.

## Run order

```bash
pip install -r requirements.txt
python train_baseline.py --kit .          # fine-tunes the baseline, writes weights/best.pt
python benchmark_speed.py --repo . --images val_images --out-csv val_pred.csv --out-json val_speed.json
python score.py --pred val_pred.csv --gt val_labels.csv --image-ids val_image_ids.csv
```

## Making your Kaggle upload

```bash
python benchmark_speed.py --repo . --images test_images --out-csv test_pred.csv --out-json test_speed.json
python make_kaggle_submission.py --pred test_pred.csv --image-ids test_image_ids.csv --out submission.csv
python check_submission.py --pred submission.csv --image-ids test_image_ids.csv
```

`submission.csv` has one row per test image: `image_id,PredictionString`. `PredictionString` lists every box as
`class score x_min y_min x_max y_max`, repeated, separated by spaces, in pixels of the original image. An image with no
detections has the single word `none` as its `PredictionString` (Kaggle rejects empty cells). Keep low-confidence boxes (down to 0.001).

## Approved pretrained weights

Only generic weights trained on COCO or ImageNet: for example the Ultralytics YOLO detection weights trained on COCO, torchvision
detection or classification weights, and timm ImageNet weights. Weights trained on any road, pavement or crack data are never allowed.
If you are not sure a set of weights is allowed, do not use it. List every pretrained weight you use in your write-up.

## Data licence and attribution

The images come from the RDD2020 dataset, licensed CC BY-NC 3.0 (https://creativecommons.org/licenses/by-nc/3.0/): use them for this
competition and for non-commercial work only. Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Omata, H., Kashiyama, T., Seto, T.,
Mraz, A. and Sekimoto, Y. (2021), "RDD2020: An Image Dataset for Smartphone-based Road Damage Detection and Classification",
Mendeley Data, V1, doi: 10.17632/5ty2wb6gvg.1. The images were renamed and re-saved, and part of the test set was degraded on purpose.
