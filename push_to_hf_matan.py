# -*- coding: utf-8 -*-
"""
Pushes the converted dataset_matan parquet shards (see to_parquet_pairs.py) to
the HuggingFace Hub as cyttic/trocr-hebrew-matan.

Usage:
    python push_to_hf_matan.py
"""

import glob
import os
from datasets import load_dataset

PARQUET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/matan_parquet")
REPO_ID = "cyttic/trocr-hebrew-matan"

data_files = {}
for split in ("train", "test"):
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, f"{split}-*.parquet")))
    if files:
        data_files[split] = files

if not data_files:
    raise FileNotFoundError(f"No parquet shards found in {PARQUET_DIR} -- run to_parquet_pairs.py first")

ds = load_dataset("parquet", data_files=data_files)

print(ds)
ds.push_to_hub(REPO_ID)
print(f"Done. -> https://huggingface.co/datasets/{REPO_ID}")
