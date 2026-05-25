import sys
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python test.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

processor = AutoImageProcessor.from_pretrained("sivan22/ResNet-finetuned-HHD")
model = AutoModelForImageClassification.from_pretrained("sivan22/ResNet-finetuned-HHD", use_safetensors=True)

# --- segmentation ---
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# dilate slightly to merge broken strokes within one character
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(binary, kernel, iterations=1)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

# filter out background (label 0) and tiny noise
min_area = 50
chars = []
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area >= min_area and h > 5 and w > 3:
        chars.append((x, y, w, h))

# Hebrew is RTL — sort right to left
chars.sort(key=lambda c: c[0], reverse=True)

avg_char_width = np.mean([w for _, _, w, _ in chars])
space_threshold = avg_char_width * 0.8

# --- classify each character ---
pad = 4
pil_img = Image.open(image_path).convert("RGB")
img_w, img_h = pil_img.size

print("Detected characters (right to left):")
results = []
for i, (x, y, w, h) in enumerate(chars):
    # insert space if gap to previous char (RTL: previous = higher x) is large
    if i > 0:
        prev_x = chars[i - 1][0]
        gap = prev_x - (x + w)
        if gap > space_threshold:
            results.append((" ", 1.0))
            print("  [SPACE]")

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + w + pad)
    y1 = min(img_h, y + h + pad)
    crop = pil_img.crop((x0, y0, x1, y1))

    inputs = processor(images=crop, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    prob = torch.softmax(logits, dim=-1)[0]
    top = torch.argmax(prob).item()
    label = model.config.id2label[top]
    confidence = prob[top].item()
    results.append((label, confidence))
    print(f"  {label}  ({confidence:.2%})")

print("\nFull text: " + "".join(r[0] for r in results))
