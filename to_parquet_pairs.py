# -*- coding: utf-8 -*-
"""
Converts a "paired sidecar" line-image set ({name}.png + {name}.gt.txt in a flat
directory, e.g. dataset_matan) into Parquet for the HuggingFace Hub.

Input:
    /mnt/ssd2/cyttic/datasets/dataset_matan/cleaned_dataset/train_pairs/
    ├── w100_F_1_form37_line_000.png
    ├── w100_F_1_form37_line_000.gt.txt
    └── ...

Output (HF split-detection convention):
    dataset/matan_parquet/
    ├── train-00000-of-0000N.parquet
    └── test-00000-of-0000N.parquet   (only written if --test-frac > 0)

Each row: {image, text, writer}. `writer` is the leading "wNNN" token parsed from
the filename -- kept so a later leak-free, writer-level split is possible (the
same principle CLAUDE.md requires for the human set: never let one writer's
lines land in both train and test).

Usage:
    # whole set as a single 'train' split (default -- push as-is, split later)
    python3 to_parquet_pairs.py

    # writer-level train/test split baked in now (leak-free, like split_dataset.py)
    python3 to_parquet_pairs.py --test-frac 0.1 --seed 42
"""

import os
import io
import glob
import random
import argparse
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage

TARGET_HEIGHT = 64       # pre-resize to BlockProcessor height to save space


def load_pairs(data_dir):
    rows = []
    for png_path in sorted(glob.glob(os.path.join(data_dir, '*.png'))):
        gt_path = os.path.splitext(png_path)[0] + '.gt.txt'
        if not os.path.exists(gt_path):
            continue
        with open(gt_path, encoding='utf-8') as f:
            text = f.read().strip()
        if not text:
            continue
        fname = os.path.basename(png_path)
        writer = fname.split('_', 1)[0]   # "w100_F_1_form37_line_000.png" -> "w100"
        rows.append({'path': png_path, 'text': text, 'writer': writer})
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


def write_split(name, rows, out_dir, shard_size):
    total = len(rows)
    if total == 0:
        print(f"  {name}: 0 rows, skipping")
        return
    n_shards = (total + shard_size - 1) // shard_size
    print(f"  {name}: {total:,} rows  ->  {n_shards} shard(s)")

    for i in range(n_shards):
        chunk = rows[i * shard_size: (i + 1) * shard_size]
        images, texts, writers = [], [], []
        for r in chunk:
            try:
                images.append(image_to_bytes(r['path']))
                texts.append(r['text'])
                writers.append(r['writer'])
            except Exception as e:
                print(f"    [WARN] skipped {r['path']}: {e}")

        ds = Dataset.from_dict(
            {'image': images, 'text': texts, 'writer': writers},
            features=Features({'image': HFImage(), 'text': Value('string'), 'writer': Value('string')}),
        )

        fname = f'{name}-{i:05d}-of-{n_shards:05d}.parquet'
        out_path = os.path.join(out_dir, fname)
        ds.to_parquet(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"    [{i+1}/{n_shards}] {out_path}  ({size_mb:.1f} MB, {len(texts)} rows)")


def main(args):
    rows = load_pairs(args.data_dir)
    print(f"found {len(rows):,} image/text pairs in {args.data_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.test_frac <= 0:
        write_split('train', rows, args.out_dir, args.shard_size)
    else:
        # writer-level split: leak-free, same principle as split_dataset.py
        # (a writer's lines never appear in both train and test)
        writers = sorted({r['writer'] for r in rows})
        rng = random.Random(args.seed)
        rng.shuffle(writers)
        n_test = max(1, round(len(writers) * args.test_frac))
        test_writers = set(writers[:n_test])

        train_rows = [r for r in rows if r['writer'] not in test_writers]
        test_rows = [r for r in rows if r['writer'] in test_writers]
        print(f"writer split: {len(writers)} writers -> "
              f"{len(writers) - n_test} train / {n_test} test (seed={args.seed})")

        write_split('train', train_rows, args.out_dir, args.shard_size)
        write_split('test', test_rows, args.out_dir, args.shard_size)

    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='/mnt/ssd2/cyttic/datasets/dataset_matan/cleaned_dataset/train_pairs',
                   help='flat dir of {name}.png + {name}.gt.txt sidecar pairs')
    p.add_argument('--out-dir', default='dataset/matan_parquet')
    p.add_argument('--shard-size', type=int, default=5_000, help='rows per parquet shard')
    p.add_argument('--test-frac', type=float, default=0.0,
                   help='fraction of WRITERS held out for test (0 = single train split, no split)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    main(args)
