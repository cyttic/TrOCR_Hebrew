# -*- coding: utf-8 -*-
"""
Random train/test split of the synthetic dataset.

The synthetic corpus has no duplicate sentences (every image renders a unique
line from sentences.txt), so a plain per-line split is leak-free — no transcript
can appear in both splits. This is the synthetic counterpart of the
document-level split used for the human set.

Source (left untouched):
    dataset/syntetic/
    ├── images/           — *.png  (uuid.png)
    └── labels.tsv        — filename<TAB>transcript

Output:
    dataset/syntetic_split/
    ├── train/
    │   ├── images/
    │   └── labels.tsv
    └── test/
        ├── images/
        └── labels.tsv

Usage:
    python split_synthetic.py [--test-frac 0.20] [--seed 42] [--move]
"""

import os
import shutil
import random
import argparse

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.join(ROOT, "dataset/syntetic")
SRC_IMAGES = os.path.join(SRC_DIR, "images")
SRC_TSV    = os.path.join(SRC_DIR, "labels.tsv")
OUT_DIR    = os.path.join(ROOT, "dataset/syntetic_split")


def load_rows(tsv_path):
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                rows.append((parts[0], parts[1]))
    return rows


def write_split(name, rows, move):
    img_dir  = os.path.join(OUT_DIR, name, "images")
    tsv_path = os.path.join(OUT_DIR, name, "labels.tsv")
    os.makedirs(img_dir, exist_ok=True)

    transfer = shutil.move if move else shutil.copy2
    out = []
    for i, (fname, text) in enumerate(rows, 1):
        src = os.path.join(SRC_IMAGES, fname)
        if not os.path.exists(src):
            print(f"  [WARN] missing image, skipped: {fname}")
            continue
        transfer(src, os.path.join(img_dir, fname))
        out.append(f"{fname}\t{text}")
        if i % 50000 == 0:
            print(f"  {name}: {i}/{len(rows)}")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return len(out)


def main(args):
    rows = load_rows(SRC_TSV)

    random.seed(args.seed)
    random.shuffle(rows)

    n_test     = max(1, round(len(rows) * args.test_frac))
    test_rows  = rows[:n_test]
    train_rows = rows[n_test:]

    if os.path.exists(OUT_DIR):
        print(f"Removing existing {OUT_DIR}/ ...")
        shutil.rmtree(OUT_DIR)

    print(f"Lines: {len(rows)}  ->  train {len(train_rows)} | test {len(test_rows)}")
    print("Writing train split...")
    n_train = write_split("train", train_rows, args.move)
    print("Writing test split...")
    n_test_lines = write_split("test", test_rows, args.move)

    # sanity: no filename appears in both splits
    train_names = {r[0] for r in train_rows}
    test_names  = {r[0] for r in test_rows}
    overlap = train_names & test_names

    print("\nDone.")
    print(f"  train: {n_train:6d} lines  ->  {OUT_DIR}/train/")
    print(f"  test:  {n_test_lines:6d} lines  ->  {OUT_DIR}/test/")
    print(f"  filename overlap between splits: {len(overlap)}  (must be 0)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--move", action="store_true",
                        help="move images instead of copying (saves disk; "
                             "empties dataset/syntetic/images/)")
    main(parser.parse_args())
