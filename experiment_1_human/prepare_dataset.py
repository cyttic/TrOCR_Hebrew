"""
Converts the SCE dataset (TIF images + JSON quad labels) into flat
images/ + labels.tsv format ready for TrOCR fine-tuning.

Usage:
    python prepare_dataset.py

Output:
    dataset/
    ├── images/   — PNG crops, one per text line
    └── labels.tsv
"""

import json
import os
import cv2
import numpy as np
from PIL import Image

SRC_IMAGES = "/mnt/ssd2/cyttic/datasets/sce_dataset/Dataset_Output/Data/Images"
SRC_LABELS = "/mnt/ssd2/cyttic/datasets/sce_dataset/Dataset_Output/Data/json_labels"
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "dataset")
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_TSV    = os.path.join(OUT_DIR, "labels.tsv")

os.makedirs(OUT_IMAGES, exist_ok=True)


def sort_quad_points(points):
    """Sort 4 points into [top-left, top-right, bottom-right, bottom-left]
    regardless of the original ordering in the JSON."""
    pts = np.array(points, dtype=np.float32)
    # split into top/bottom by y value
    top2    = pts[np.argsort(pts[:, 1])[:2]]
    bottom2 = pts[np.argsort(pts[:, 1])[2:]]
    # within each pair sort by x
    tl, tr = top2[np.argsort(top2[:, 0])]
    bl, br = bottom2[np.argsort(bottom2[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_quad(img_np, points):
    """Perspective-warp a quadrilateral region to a straight rectangle."""
    pts = sort_quad_points(points)
    tl, tr, br, bl = pts

    width  = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    if width < 1 or height < 1:
        return None

    dst = np.array([
        [0,         0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0,         height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img_np, M, (width, height))


json_files = sorted(f for f in os.listdir(SRC_LABELS) if f.endswith(".json"))

total = 0
skipped = 0
rows = []

for json_file in json_files:
    base = os.path.splitext(json_file)[0]
    img_path = os.path.join(SRC_IMAGES, base + ".tif")

    if not os.path.exists(img_path):
        print(f"  [WARN] image not found: {img_path}")
        continue

    with open(os.path.join(SRC_LABELS, json_file)) as f:
        data = json.load(f)

    lines = [s for s in data["shapes"] if s.get("type") == "line"]
    labeled = [l for l in lines if l.get("transcript", "").strip()]

    if not labeled:
        continue

    img = np.array(Image.open(img_path).convert("RGB"))

    for line in labeled:
        transcript = line["transcript"].strip()
        idx        = line["line_index"]
        points     = line["points"]

        crop = warp_quad(img, points)
        if crop is None:
            skipped += 1
            continue

        filename = f"{base}_line{idx:03d}.png"
        out_path = os.path.join(OUT_IMAGES, filename)
        cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        rows.append(f"{filename}\t{transcript}")
        total += 1

    print(f"  {base}: {len(labeled)} lines")

with open(OUT_TSV, "w", encoding="utf-8") as f:
    f.write("\n".join(rows))

print(f"\nDone.")
print(f"  Saved:   {total} line crops -> {OUT_IMAGES}/")
print(f"  Labels:  {OUT_TSV}")
print(f"  Skipped: {skipped} (degenerate quads)")
