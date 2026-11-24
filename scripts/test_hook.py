import albumentations as A
from ultralytics.data import augment

orig_init = augment.Albumentations.__init__

def custom_init(self, p=1.0, transforms=None, flip_idx=None):
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
    return orig_init(self, p=p, transforms=transforms, flip_idx=flip_idx)

augment.Albumentations.__init__ = custom_init
alb = augment.Albumentations()
print("Custom Albumentations hooked successfully! Loaded transforms count:", len(alb.transform.transforms))
