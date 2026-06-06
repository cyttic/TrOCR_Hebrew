from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={
        "train": "/mnt/ssd2/cyttic/projects/TrOCR_Hebrew/dataset/syntetic_parquet/train-*.parquet",
        "test":  "/mnt/ssd2/cyttic/projects/TrOCR_Hebrew/dataset/syntetic_parquet/test-*.parquet",
    }
)

print(ds)
ds.push_to_hub("cyttic/trocr-hebrew-synthetic")
print("Done.")
