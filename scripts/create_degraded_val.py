import os
import random
import cv2
import numpy as np
import albumentations as A

def create_degraded_val_set(val_dir="val_images", out_dir="val_images_degraded", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    
    # 4 distinct corruptions matching competition stress suite
    t_blur = A.OneOf([
        A.MotionBlur(blur_limit=(7, 15), p=1.0),
        A.GaussianBlur(blur_limit=(7, 15), p=1.0),
    ], p=1.0)
    
    t_lowlight = A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=(-0.45, -0.25), contrast_limit=(-0.35, -0.15), p=1.0),
        A.RandomGamma(gamma_limit=(50, 75), p=1.0),
    ], p=1.0)
    
    t_noise = A.OneOf([
        A.GaussNoise(std_range=(0.15, 0.35), p=1.0),
        A.ISONoise(color_shift=(0.05, 0.15), intensity=(0.4, 0.8), p=1.0),
    ], p=1.0)
    
    t_jpeg = A.ImageCompression(quality_range=(10, 25), p=1.0)
    
    transforms = [t_blur, t_lowlight, t_noise, t_jpeg]
    type_names = ["Blur (Motion/Gaussian)", "Low-Light (Brightness/Gamma)", "Noise (Gauss/ISO)", "Heavy JPEG (q10-25)"]
    
    img_files = sorted([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total = len(img_files)
    counts = [0, 0, 0, 0]
    
    print(f"Generating degraded validation set from {total} images in {val_dir}...")
    
    for i, fname in enumerate(img_files):
        src_path = os.path.join(val_dir, fname)
        dst_path = os.path.join(out_dir, fname)
        
        img = cv2.imread(src_path)
        if img is None:
            continue
        
        # Cycle through the 4 degradation types deterministically
        choice = i % 4
        counts[choice] += 1
        degraded = transforms[choice](image=img)["image"]
        
        # Compound degradation for 15% of images
        if random.random() < 0.15:
            sec_choice = random.choice([c for c in range(4) if c != choice])
            degraded = transforms[sec_choice](image=degraded)["image"]
            
        cv2.imwrite(dst_path, degraded)
        
    print(f"Successfully generated {total} images in {out_dir}/:")
    for name, cnt in zip(type_names, counts):
        print(f"  - {name}: {cnt} images ({cnt/total*100:.1f}%)")

if __name__ == "__main__":
    create_degraded_val_set()
