# -*- coding: utf-8 -*-
"""
Converts the document-level train/test split into Parquet for HuggingFace Hub.

Reads the directories produced by split_dataset.py and writes HF-convention
parquet files so `load_dataset("<user>/<repo>")` returns proper train/test
splits automatically.

Input:
    dataset/human_split/
    ├── train/ {images/, labels.tsv}
    └── test/  {images/, labels.tsv}

Output:
    dataset/human_parquet/
    ├── train-00000-of-00001.parquet
    └── test-00000-of-00001.parquet

Usage:
    python3 to_parquet.py
"""

import os
import io
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage

SPLITS = {
    'train': ('dataset/human_split/train/labels.tsv', 'dataset/human_split/train/images'),
    'test':  ('dataset/human_split/test/labels.tsv',  'dataset/human_split/test/images'),
}
OUT_DIR = 'dataset/human_parquet'

SHARD_SIZE    = 5_000    # rows per Parquet shard
TARGET_HEIGHT = 64       # pre-resize to BlockProcessor height to save space


def load_tsv(tsv_path):
    rows = []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2 and parts[0] and parts[1]:
                rows.append({'file': parts[0], 'text': parts[1]})
    return rows


def image_to_bytes(img_path):
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        w, h = img.size
        new_w = max(1, round(w * TARGET_HEIGHT / h))
        img = img.resize((new_w, TARGET_HEIGHT), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()


def convert_split(name, tsv_path, img_dir):
    rows = load_tsv(tsv_path)
    total = len(rows)
    n_shards = (total + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"  {name}: {total:,} rows  ->  {n_shards} shard(s)")

    for i in range(n_shards):
        chunk = rows[i * SHARD_SIZE : (i + 1) * SHARD_SIZE]
        images, texts, docs = [], [], []
        for r in chunk:
            path = os.path.join(img_dir, r['file'])
            if not os.path.exists(path):
                continue
            try:
                images.append(image_to_bytes(path))
                texts.append(r['text'])
                # source document = filename without the _line{NNN}.png suffix
                docs.append(r['file'].rsplit('_line', 1)[0])
            except Exception as e:
                print(f"    [WARN] skipped {r['file']}: {e}")

        ds = Dataset.from_dict(
            {'image': images, 'text': texts, 'source_doc': docs},
            features=Features({
                'image': HFImage(),
                'text': Value('string'),
                'source_doc': Value('string'),
            })
        )

        # HuggingFace split-detection convention: {split}-{i}-of-{n}.parquet
        fname = f'{name}-{i:05d}-of-{n_shards:05d}.parquet'
        out_path = os.path.join(OUT_DIR, fname)
        ds.to_parquet(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"    [{i+1}/{n_shards}] {out_path}  ({size_mb:.1f} MB, {len(texts)} rows)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, (tsv_path, img_dir) in SPLITS.items():
        if not os.path.exists(tsv_path) or not os.path.isdir(img_dir):
            print(f"Skipping {name}: {tsv_path} or {img_dir} not found.")
            continue
        convert_split(name, tsv_path, img_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
