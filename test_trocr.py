import sys
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from PIL import Image
import torch
import cv2
import numpy as np

from block_processor import HebrewBlockProcessor

if len(sys.argv) < 2:
    print("Usage: python test_trocr.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

processor = HebrewBlockProcessor()
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert")
model = VisionEncoderDecoderModel.from_pretrained("trocr-hebrew-untrained")

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.sep_token_id

# --- line segmentation ---
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

h_proj = np.sum(binary, axis=1)

in_line = False
lines = []
start = 0
for i, val in enumerate(h_proj):
    if val > 0 and not in_line:
        in_line = True
        start = i
    elif val == 0 and in_line:
        in_line = False
        lines.append((start, i))
if in_line:
    lines.append((start, len(h_proj)))

pil_img = Image.open(image_path).convert("RGB")
img_w, img_h = pil_img.size

pad = 4
print(f"Detected {len(lines)} line(s):\n")

for i, (y0, y1) in enumerate(lines):
    crop = pil_img.crop((0, max(0, y0 - pad), img_w, min(img_h, y1 + pad)))
    inputs = processor(images=crop, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            inputs["pixel_values"],
            num_beams=4,
            length_penalty=0.5,
            max_new_tokens=64,
        )
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    print(f"Line {i + 1}: {text}")
