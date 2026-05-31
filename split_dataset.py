# -*- coding: utf-8 -*-
"""
Document-level train/test split of the human handwritten dataset.

Whole documents go entirely to train OR test — no document (and therefore no
writer's copy of a sentence) appears in both splits. This avoids the leakage
that a naive per-line split would cause, since the dataset is ~280 writers
copying the same ~127 sentences.

Source (left untouched):
    dataset/human/
    ├── images/           — *.png  ({doc}_line{NNN}.png)
    └── labels.tsv        — filename<TAB>transcript

Output:
    dataset/human_split/
    ├── train/
    │   ├── images/
    │   └── labels.tsv
    └── test/
        ├── images/
        └── labels.tsv

Usage:
    python split_dataset.py [--test-frac 0.10] [--seed 42]
"""

import os
import re
import shutil
import random
import argparse
from collections import defaultdict

SRC_DIR    = "dataset/human"
SRC_IMAGES = os.path.join(SRC_DIR, "images")
SRC_TSV    = os.path.join(SRC_DIR, "labels.tsv")
OUT_DIR    = "dataset/human_split"

DOC_RE = re.compile(r"_line\d+\.png$")


def document_of(filename):
    """BRN..._0000000100_line002.png  ->  BRN..._0000000100"""
    return DOC_RE.sub("", filename)


def load_rows(tsv_path):
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                rows.append((parts[0], parts[1]))
    return rows


def write_split(name, rows):
    img_dir = os.path.join(OUT_DIR, name, "images")
    tsv_path = os.path.join(OUT_DIR, name, "labels.tsv")
    os.makedirs(img_dir, exist_ok=True)

    out = []
    for fname, text in rows:
        src = os.path.join(SRC_IMAGES, fname)
        if not os.path.exists(src):
            print(f"  [WARN] missing image, skipped: {fname}")
            continue
        shutil.copy2(src, os.path.join(img_dir, fname))
        out.append(f"{fname}\t{text}")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return len(out)


def main(args):
    rows = load_rows(SRC_TSV)

    # group rows by document
    by_doc = defaultdict(list)
    for fname, text in rows:
        by_doc[document_of(fname)].append((fname, text))

    docs = sorted(by_doc.keys())          # sort first for determinism
    random.seed(args.seed)
    random.shuffle(docs)

    n_test = max(1, round(len(docs) * args.test_frac))
    test_docs  = set(docs[:n_test])
    train_docs = set(docs[n_test:])
    assert not (test_docs & train_docs), "document overlap detected!"

    train_rows = [r for d in train_docs for r in by_doc[d]]
    test_rows  = [r for d in test_docs  for r in by_doc[d]]

    if os.path.exists(OUT_DIR):
        print(f"Removing existing {OUT_DIR}/ ...")
        shutil.rmtree(OUT_DIR)

    print(f"Documents: {len(docs)}  ->  train {len(train_docs)} | test {len(test_docs)}")
    print("Copying train split...")
    n_train = write_split("train", train_rows)
    print("Copying test split...")
    n_test_lines = write_split("test", test_rows)

    # sanity: no document leaks across splits
    train_doc_check = {document_of(r[0]) for r in train_rows}
    test_doc_check  = {document_of(r[0]) for r in test_rows}
    overlap = train_doc_check & test_doc_check

    print("\nDone.")
    print(f"  train: {len(train_docs):3d} docs | {n_train:5d} lines  ->  {OUT_DIR}/train/")
    print(f"  test:  {len(test_docs):3d} docs | {n_test_lines:5d} lines  ->  {OUT_DIR}/test/")
    print(f"  document overlap between splits: {len(overlap)}  (must be 0)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed",      type=int,   default=42)
    main(parser.parse_args())
