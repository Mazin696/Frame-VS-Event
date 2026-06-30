# csv_roi_to_images.py
# Convert ROI-only CSVs -> grayscale images for Edge Impulse.

import os, csv
from glob import glob
from pathlib import Path
import numpy as np
from PIL import Image

INPUT_ROOT = r"C:\Users\Mazin Abdallah\Desktop\Event-Recognition\Events"
   # copy from SD to PC first
OUTPUT_ROOT = r"C:\Users\Mazin Abdallah\Desktop\Event-Recognition\images"
OUT_SIZE    = (96, 96)             # EI input size (grayscale)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def rasterize_csv(csv_path, W=320, H=320):
    grid = np.zeros((H, W), dtype=np.int32)
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            x = int(row["x"]); y = int(row["y"])
            if 0 <= x < W and 0 <= y < H:
                grid[y, x] += 1  # unsigned density
    return grid

def normalize_to_u8(arr):
    mn, mx = int(arr.min()), int(arr.max())
    if mx == mn:
        return np.full_like(arr, 128, dtype=np.uint8)
    arr = arr.astype(np.float32)
    arr = (arr - mn) / (mx - mn)
    return (255.0 * arr).astype(np.uint8)

def process_one(csv_path):
    grid = rasterize_csv(csv_path, 320, 320)
    img_u8 = normalize_to_u8(grid)
    pil = Image.fromarray(img_u8, mode="L").resize(OUT_SIZE, Image.BILINEAR)
    return pil

def convert_all():
    ensure_dir(OUTPUT_ROOT)
    for class_dir in sorted(glob(os.path.join(INPUT_ROOT, "*"))):
        if not os.path.isdir(class_dir):
            continue
        label = os.path.basename(class_dir)
        out_dir = os.path.join(OUTPUT_ROOT, label)
        ensure_dir(out_dir)
        for csv_path in sorted(glob(os.path.join(class_dir, "*.csv"))):
            out_path = os.path.join(out_dir, Path(csv_path).stem + ".jpg")
            pil = process_one(csv_path)
            pil.save(out_path, quality=95)
            print("wrote:", out_path)

if __name__ == "__main__":
    convert_all()
