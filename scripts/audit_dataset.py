import os
import csv
from collections import Counter, defaultdict
from PIL import Image
import numpy as np

def audit_split(split_name, img_dir, lbl_csv):
    print(f"=== Auditing {split_name} ===")
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_imgs = len(img_files)
    
    # Load labels
    boxes_by_img = defaultdict(list)
    class_counts = Counter()
    box_widths = []
    box_heights = []
    box_areas = []
    class_box_stats = defaultdict(lambda: {"w": [], "h": [], "area": [], "aspect": []})
    
    with open(lbl_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            iid = r["image_id"]
            cid = int(r["class_id"])
            x1 = float(r["x_min"])
            y1 = float(r["y_min"])
            x2 = float(r["x_max"])
            y2 = float(r["y_max"])
            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect = w / max(h, 1e-5)
            
            boxes_by_img[iid].append((cid, x1, y1, x2, y2))
            class_counts[cid] += 1
            box_widths.append(w)
            box_heights.append(h)
            box_areas.append(area)
            class_box_stats[cid]["w"].append(w)
            class_box_stats[cid]["h"].append(h)
            class_box_stats[cid]["area"].append(area)
            class_box_stats[cid]["aspect"].append(aspect)
            
    total_boxes = sum(class_counts.values())
    images_with_boxes = len(boxes_by_img)
    undamaged_imgs = total_imgs - images_with_boxes
    pct_undamaged = (undamaged_imgs / total_imgs) * 100
    
    print(f"Total Images: {total_imgs}")
    print(f"Images with damage: {images_with_boxes} ({images_with_boxes/total_imgs*100:.2f}%)")
    print(f"Undamaged (empty) images: {undamaged_imgs} ({pct_undamaged:.2f}%)")
    print(f"Total Bounding Boxes: {total_boxes}")
    print(f"Average boxes per damaged image: {total_boxes / max(images_with_boxes, 1):.2f}")
    
    class_names = {0: "D00 (Longitudinal)", 1: "D10 (Transverse)", 2: "D20 (Alligator)", 3: "D40 (Pothole)"}
    print("\nClass Distribution:")
    for cid in range(4):
        c_count = class_counts[cid]
        c_pct = (c_count / max(total_boxes, 1)) * 100
        print(f"  Class {cid} - {class_names[cid]}: {c_count} ({c_pct:.2f}%)")
        
    print("\nBounding Box Dimensions (pixels):")
    print(f"  Width  : Mean={np.mean(box_widths):.1f}, Median={np.median(box_widths):.1f}, Min={np.min(box_widths):.1f}, Max={np.max(box_widths):.1f}")
    print(f"  Height : Mean={np.mean(box_heights):.1f}, Median={np.median(box_heights):.1f}, Min={np.min(box_heights):.1f}, Max={np.max(box_heights):.1f}")
    print(f"  Area   : Mean={np.mean(box_areas):.1f}, Median={np.median(box_areas):.1f}, Min={np.min(box_areas):.1f}, Max={np.max(box_areas):.1f}")
    
    # Resolutions
    print("\nSampling Image Resolutions (first 500 images):")
    res_counter = Counter()
    for p in img_files[:500]:
        with Image.open(os.path.join(img_dir, p)) as im:
            res_counter[im.size] += 1
    for res, count in res_counter.most_common(5):
        print(f"  Resolution {res[0]}x{res[1]}: {count} images ({count/min(500, total_imgs)*100:.1f}%)")
    print()

audit_split("TRAIN", "train_images", "train_labels.csv")
audit_split("VALIDATION", "val_images", "val_labels.csv")
