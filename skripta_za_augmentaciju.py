import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

INPUT_ROOT = Path(r"C:\Users\Korisnik\Desktop\dataset\LEGO_GAN")

OUTPUT_ROOT = INPUT_ROOT.parent / "LEGO_GAN_AUG"

TARGET_PER_CLASS = 500   

RESIZE_TO = 128

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(42)
np.random.seed(42)

# 2) DEFINICIJA AUGMENTACIJA

augmenter = A.Compose([
   A.RandomResizedCrop(
    size=(RESIZE_TO, RESIZE_TO),
    scale=(0.75, 1.0),
    ratio=(0.9, 1.1),
    p=0.5
),
    A.ShiftScaleRotate(
        shift_limit=0.06,
        scale_limit=0.12,
        rotate_limit=15,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.6
    ),
    A.Perspective(scale=(0.02, 0.06), keep_size=True, p=0.2),
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.45
    ),
    A.HueSaturationValue(
        hue_shift_limit=8,
        sat_shift_limit=12,
        val_shift_limit=8,
        p=0.3
    ),
    A.Sharpen(alpha=(0.1, 0.25), lightness=(0.9, 1.1), p=0.2),
    A.GaussianBlur(blur_limit=(3, 5), p=0.12),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    A.HorizontalFlip(p=0.5),
], p=1.0)


def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in EXTS]

def read_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def ensure_size(img):
    if img.shape[0] != RESIZE_TO or img.shape[1] != RESIZE_TO:
        img = cv2.resize(img, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)
    return img

def write_rgb(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = ensure_size(img)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    classes = [
        d for d in INPUT_ROOT.iterdir()
        if d.is_dir() and d.name not in {"test"}
    ]

    print("Klase:", [c.name for c in classes])
    print("Output folder:", OUTPUT_ROOT)

    for cls in classes:
        print(f"\n=== Obrada klase: {cls.name} ===")
        out_dir = OUTPUT_ROOT / cls.name
        out_dir.mkdir(parents=True, exist_ok=True)

        originals = list_images(cls)

        # 1) kopiraj originale
        for p in originals:
            img = read_rgb(p)
            if img is None:
                continue
            write_rgb(out_dir / f"{p.stem}_orig.jpg", img)

        current = len(list_images(out_dir))
        need = max(0, TARGET_PER_CLASS - current)

        # 2) augmentacije
        for i in tqdm(range(need), desc="Augmentacija"):
            src = random.choice(originals)
            img = read_rgb(src)
            if img is None:
                continue
            img = ensure_size(img)
            aug = augmenter(image=img)["image"]
            write_rgb(out_dir / f"{src.stem}_aug_{i:05d}.jpg", aug)

        print(f"Završeno: {cls.name} → {len(list_images(out_dir))} slika")

    print("\nGOTOVO. Proširen dataset je u:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
