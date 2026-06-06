# Experiment 2 — Synthetic data pretrain

Stage-1 of the correct recipe (**synthetic pretrain → human finetune**): render
synthetic Hebrew line images, pretrain the TrOCR model on them, then continue for
more epochs. Produces `cyttic/trocr-hebrew-synthetic` (dataset) and the
`trocr-hebrew-synthetic-cont` model.

## Files
| File | Purpose |
|------|---------|
| `generate_synthetic.py` | Render synthetic Hebrew line images → `dataset/syntetic/` (clean render, no aug) |
| `split_synthetic.py` | **Per-line** leak-free split → `dataset/syntetic_split/` |
| `train_trocr_synthetic.ipynb` | Stage-1 synthetic pretrain (Hub-checkpointed, auto-resume) |
| `continue_synthetic_train.py` | Warm-started continuation (more epochs), GCS-checkpointed |
| `kaggle_eval_full.py` | Full synthetic-test CER/WER/BLEU eval on a Kaggle T4 |
| `push_to_hf.py` | Push the synthetic parquet dataset to the Hub |
| `build_report.py` | Build the `.docx` training report + `training_curve.png` |
| `eval_results.md` / `training_config.md` | Logged results & hyperparameters |
| `TrOCR_Hebrew_training_report.docx`, `training_curve.png` | Generated report artifacts |
| `models/trocr-hebrew-synthetic-cont/` | The continued-pretrain model from this experiment |

## Results so far
Best: **CER ~0.415** (beam=4, 2k synthetic test samples) — still undertrained
(usable reader wants CER < ~0.10). Biggest remaining lever: **unfreeze the
encoder** for the next continuation. See `eval_results.md`.

## Notes
- Shared code/data stays at the **repo root** (`block_processor.py`,
  `to_parquet.py`, `dataset/`, etc.). Scripts resolve root paths via a `ROOT`
  anchor, so they run from this folder regardless of cwd.
- `generate_synthetic.py` uses absolute paths for its font/sentence corpora.
