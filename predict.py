# -*- coding: utf-8 -*-
"""
Run the fine-tuned Hebrew TrOCR model on handwritten line images.
Works on CPU (no GPU required).

Needs `block_processor.py` next to this file (the custom HebrewBlockProcessor).

Install deps once:
    pip install torch transformers pillow

Usage:
    python predict.py --model my_model.zip --image line.png
    python predict.py --model my_model     --image folder_of_lines/
    python predict.py --model my_model     --image line.png --beams 4   # higher quality
"""

import os
import glob
import zipfile
import argparse

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer

from block_processor import HebrewBlockProcessor


def resolve_model_dir(path):
    """Accept either a model directory or a .zip; return the dir holding config.json."""
    if path.endswith(".zip"):
        out = path[:-4] + "_extracted"
        if not os.path.isdir(out):
            print(f"Extracting {path} -> {out}/")
            with zipfile.ZipFile(path) as z:
                z.extractall(out)
        if os.path.exists(os.path.join(out, "config.json")):
            return out
        for root, _, files in os.walk(out):          # model may sit in a subfolder
            if "config.json" in files:
                return root
        return out
    return path


def collect_images(path):
    if os.path.isdir(path):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
            files += glob.glob(os.path.join(path, ext))
        return sorted(files)
    return [path]


def main(args):
    model_dir = resolve_model_dir(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_dir} on {device} ...")
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = HebrewBlockProcessor()

    # make sure generation has its special tokens (safe even if already set)
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id
    model.generation_config.max_new_tokens = None

    images = collect_images(args.image)
    if not images:
        print(f"No images found at: {args.image}")
        return
    print(f"{len(images)} image(s)\n" + "-" * 60)

    for path in images:
        img = Image.open(path).convert("RGB")
        pv = processor([img])["pixel_values"].to(device)
        with torch.no_grad():
            ids = model.generate(pv, num_beams=args.beams, max_length=args.max_length)
        text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        print(f"{os.path.basename(path)}\n  -> {text}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model directory or .zip file")
    p.add_argument("--image", required=True, help="image file or a folder of images")
    p.add_argument("--beams", type=int, default=1, help="1 = greedy/fast, 4 = beam/better")
    p.add_argument("--max-length", type=int, default=128)
    main(p.parse_args())
