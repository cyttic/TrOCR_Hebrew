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
    # human set (default): document split, source_doc column, 5k rows/shard
    python3 to_parquet.py

    # synthetic set: no source_doc, larger shards
    python3 to_parquet.py --split-dir dataset/syntetic_split \
        --out-dir dataset/syntetic_parquet --shard-size 20000
"""

import os
import io
import argparse
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage

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


def convert_split(name, tsv_path, img_dir, out_dir, shard_size, with_source_doc):
    rows = load_tsv(tsv_path)
    total = len(rows)
    n_shards = (total + shard_size - 1) // shard_size
    print(f"  {name}: {total:,} rows  ->  {n_shards} shard(s)")

    for i in range(n_shards):
        chunk = rows[i * shard_size : (i + 1) * shard_size]
        images, texts, docs = [], [], []
        for r in chunk:
            path = os.path.join(img_dir, r['file'])
            if not os.path.exists(path):
                continue
            try:
                images.append(image_to_bytes(path))
                texts.append(r['text'])
                if with_source_doc:
                    # source document = filename without the _line{NNN}.png suffix
                    docs.append(r['file'].rsplit('_line', 1)[0])
            except Exception as e:
                print(f"    [WARN] skipped {r['file']}: {e}")

        cols  = {'image': images, 'text': texts}
        feats = {'image': HFImage(), 'text': Value('string')}
        if with_source_doc:
            cols['source_doc']  = docs
            feats['source_doc'] = Value('string')
        ds = Dataset.from_dict(cols, features=Features(feats))

        # HuggingFace split-detection convention: {split}-{i}-of-{n}.parquet
        fname = f'{name}-{i:05d}-of-{n_shards:05d}.parquet'
        out_path = os.path.join(out_dir, fname)
        ds.to_parquet(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"    [{i+1}/{n_shards}] {out_path}  ({size_mb:.1f} MB, {len(texts)} rows)")


def main(args):
    splits = {
        'train': (os.path.join(args.split_dir, 'train/labels.tsv'),
                  os.path.join(args.split_dir, 'train/images')),
        'test':  (os.path.join(args.split_dir, 'test/labels.tsv'),
                  os.path.join(args.split_dir, 'test/images')),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    for name, (tsv_path, img_dir) in splits.items():
        if not os.path.exists(tsv_path) or not os.path.isdir(img_dir):
            print(f"Skipping {name}: {tsv_path} or {img_dir} not found.")
            continue
        convert_split(name, tsv_path, img_dir,
                      args.out_dir, args.shard_size, args.source_doc)
    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split-dir',  default='dataset/human_split',
                   help='dir with train/ and test/ subdirs (default: human)')
    p.add_argument('--out-dir',    default='dataset/human_parquet')
    p.add_argument('--shard-size', type=int, default=5_000,
                   help='rows per parquet shard')
    p.add_argument('--source-doc', action='store_true', default=None,
                   help='add source_doc column (auto-on for the human default)')
    args = p.parse_args()
    # default behaviour: human set keeps its source_doc column
    if args.source_doc is None:
        args.source_doc = (args.split_dir == 'dataset/human_split')
    main(args)
