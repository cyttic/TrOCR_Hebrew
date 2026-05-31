# -*- coding: utf-8 -*-
"""
Generates 1M synthetic Hebrew text line images for TrOCR fine-tuning.

Output:
    dataset/syntetic/
    ├── images/      — PNG line crops
    └── labels.tsv   — filename → transcript

Usage:
    python generate_synthetic.py [--count 1000000] [--processes 8]
"""

import os
import sys
import random
import uuid
import argparse
import math
import multiprocessing
from multiprocessing import Value, Lock
import ctypes

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
import cv2

sys.path.insert(0, '/mnt/ssd2/cyttic/projects/handwrittenTextGenerator')
from image_distortion import apply_random_transform

# ── paths ────────────────────────────────────────────────────────────────────
FONTS_DIR      = '/mnt/ssd2/cyttic/projects/fontsVisualizer/fonts'
SENTENCES_FILE = '/mnt/ssd2/cyttic/projects/textGrabber/sentences.txt'
OUT_DIR  = '/mnt/ssd2/cyttic/projects/TrOCR_Hebrew/dataset/syntetic'
IMG_DIR  = os.path.join(OUT_DIR, 'images')
TSV_PATH = os.path.join(OUT_DIR, 'labels.tsv')

# ── corpus ───────────────────────────────────────────────────────────────────
def load_sentences():
    with open(SENTENCES_FILE, encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(sentences):,} sentences from {SENTENCES_FILE}")
    return sentences


# ── font helpers ─────────────────────────────────────────────────────────────
def get_fonts(fonts_dir):
    fonts = []
    for fname in os.listdir(fonts_dir):
        if fname.lower().endswith('.ttf') or fname.lower().endswith('.otf'):
            fonts.append(os.path.join(fonts_dir, fname))
    return fonts


def font_supports_text(font_path, text):
    try:
        from fontTools.ttLib import TTFont
        tt = TTFont(font_path)
        cmap = tt['cmap'].getBestCmap()
        return all(ord(c) in cmap for c in text if c.strip())
    except Exception:
        return True  # assume ok if check fails


# ── image generation ─────────────────────────────────────────────────────────
def render_line(text, font_path, font_size):
    font = ImageFont.truetype(font=font_path, size=font_size)
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + 10
    h = bbox[3] - bbox[1] + 10
    if w < 1 or h < 1:
        return None
    img = Image.new('RGB', (w, h), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0] + 5, -bbox[1] + 5), text, fill='black', font=font)
    return img


def augment(img):
    # blur
    if random.random() < 0.3:
        r = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=r))
    # distort — convert to grayscale uint8, distort, convert back
    if random.random() < 0.3:
        try:
            gray = np.array(img.convert('L'))
            distorted, _ = apply_random_transform(gray)
            img = Image.fromarray(distorted).convert('RGB')
        except Exception:
            pass  # skip distortion if it fails for this image
    return img


def generate_batch(fonts, sentences, out_dir,
                   shared_counter, lock, seed):
    random.seed(seed)
    np.random.seed(seed)

    rows = []

    for text in sentences:
        text = text.strip()
        if not text:
            continue

        font_path = random.choice(fonts)
        font_size = random.randint(30, 60)

        try:
            img = render_line(text, font_path, font_size)
        except Exception:
            continue
        if img is None:
            continue

        # skip lines too wide for BlockProcessor (max 2304px at 64px height)
        w, h = img.size
        scaled_w = int(w / h * 64)
        if scaled_w > 2304:
            continue

        img = augment(img)

        fname = uuid.uuid4().hex + '.png'
        img.save(os.path.join(out_dir, fname))
        rows.append(f"{fname}\t{text}")

        with lock:
            shared_counter.value += 1

    return rows


def worker(fonts, sentences, out_dir,
           shared_counter, lock, seed, result_queue):
    rows = generate_batch(fonts, sentences, out_dir,
                          shared_counter, lock, seed)
    result_queue.put(rows)


# ── main ─────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(IMG_DIR, exist_ok=True)

    sentences = load_sentences()
    fonts = get_fonts(FONTS_DIR)

    if args.count > len(sentences):
        print(f"Warning: requested {args.count:,} images but sentences.txt "
              f"only contains {len(sentences):,} sentences. "
              f"Will generate {len(sentences):,} images only.")
        args.count = len(sentences)

    # Shuffle once, then split evenly across processes — no sentence used twice
    random.shuffle(sentences)
    sentences = sentences[:args.count]
    chunk_size = math.ceil(len(sentences) / args.processes)
    chunks = [sentences[i * chunk_size:(i + 1) * chunk_size]
              for i in range(args.processes)]

    print(f"Sentences: {len(sentences):,} | Fonts: {len(fonts)} | "
          f"Processes: {args.processes} (~{chunk_size:,} each)")

    shared_counter = Value(ctypes.c_int, 0)
    lock = Lock()
    result_queue = multiprocessing.Queue()

    processes = [
        multiprocessing.Process(
            target=worker,
            args=(fonts, chunks[i], IMG_DIR,
                  shared_counter, lock, i, result_queue)
        )
        for i in range(args.processes)
    ]

    print(f"Starting {args.processes} processes ({chunk_size:,} sentences each)...")
    for p in processes:
        p.start()

    all_rows = []
    for _ in processes:
        all_rows.extend(result_queue.get())

    for p in processes:
        p.join()

    # trim to exact count
    all_rows = all_rows[:args.count]
    with open(TSV_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_rows))

    print(f"\nDone.")
    print(f"  Images: {len(all_rows):,}  →  {IMG_DIR}/")
    print(f"  Labels: {TSV_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count',     type=int, default=1_000_000)
    parser.add_argument('--processes', type=int, default=8)
    main(parser.parse_args())
