# Experiment 1 — Human dataset finetune

Fine-tuning the Hebrew TrOCR model directly on the **human** handwritten set
(`cyttic/trocr-hebrew-human`, 280 writers × 127 pangram sentences). This is the
"sentence-recognition" baseline / pipeline smoke-test described in the root
`CLAUDE.md` — useful as a baseline, **not** a general reader (every document
shares the same 127 sentences, so it learns to recite known sentences).

## Files
| File | Purpose |
|------|---------|
| `prepare_dataset.py` | SCE dataset (TIF + JSON quads) → `dataset/human/{images,labels.tsv}` |
| `split_dataset.py` | **Document-level** train/test split → `dataset/human_split/` (leak-safe) |
| `train.py` | Fine-tune (Seq2SeqTrainer), CLI flags; best model → `output/trocr-hebrew-human/best/` |
| `stage2_human_finetune.py` | Human finetune warm-started from the stage-1 synthetic model |
| `train_trocr_hebrew.ipynb` | Interactive version of the human finetune |
| `predict.py` | Run a trained model on line images |
| `test.py` / `test_trocr.py` | Ad-hoc segmentation + recognition probes |
| `models/first_iteration/` | The human-finetuned model from this experiment |

## Notes
- Shared code/data stays at the **repo root**: `block_processor.py`,
  `build_hebrew_trocr.py`, `to_parquet.py`, `eval_metrics.py`,
  `trocr-hebrew-untrained/`, `dataset/`.
- The scripts resolve those root paths automatically (via a `ROOT` anchor), so
  they run from this folder regardless of the current working directory.
