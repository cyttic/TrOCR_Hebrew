# Experiment 3 — Synthetic continuation with the encoder UNFROZEN

Continues training the **Experiment-2 model** on the **same synthetic dataset**,
but this time the ViT encoder is **unfrozen** so it can adapt to the text.

| Setting | Value |
|---------|-------|
| Warm-start model | `cyttic/trocr-hebrew-synthetic-cont` (Experiment-2 result) |
| Dataset | `cyttic/trocr-hebrew-synthetic` (same as Experiment 2) |
| Epochs | **2** |
| Encoder | **unfrozen** (full model trains) |
| Learning rate | `3e-5` (lower than the frozen 5e-5 runs; unfrozen encoder is more sensitive) |
| Batch size | 8 (unfrozen ⇒ more VRAM than the frozen batch-16 runs) |
| Precision | bf16 (L4) / fp16 (RTX 2080) |
| Output model | `cyttic/trocr-hebrew-synthetic-cont-unfrozen` |

## Why
The frozen-encoder runs plateaued at **CER ~0.41** — only the decoder + the
randomly-initialized cross-attention learned, against *fixed* visual features.
Unfreezing the ViT is the biggest remaining lever (see
`../experiment_2_synthetic/eval_results.md`).

## Files
| File | Purpose |
|------|---------|
| `train_trocr_synthetic_unfrozen.ipynb` | The experiment notebook (warm-start → 2 epochs unfrozen → eval → push) |
| `block_processor.py` | Copy of the shared `HebrewBlockProcessor` (the notebook also inlines it) |

## How to run
1. Open the notebook on a GPU box (L4 recommended) and run top to bottom.
2. It downloads the warm-start model + dataset from the Hub, so no local data is
   needed. A HuggingFace **write token** is required (auth cell).
3. Checkpoints push to `cyttic/trocr-hebrew-synthetic-cont-unfrozen-ckpts` every
   `SAVE_STEPS`; if the session dies, re-run top-to-bottom to resume.
4. The final model is saved locally and pushed to
   `cyttic/trocr-hebrew-synthetic-cont-unfrozen`.

## Knobs to watch
- `LR` — if loss/CER destabilizes early (unfrozen encoders can), drop to `2e-5`.
  If it's learning fine but slow, `5e-5`.
- `BATCH_SIZE` — drop to 4 on OOM, raise to 12 if you have VRAM headroom.
- `MAX_STEPS = 50` in the config cell gives a quick smoke test before the full run.

## Evaluation
For full-test CER/WER/BLEU, reuse `../experiment_2_synthetic/kaggle_eval_full.py`
with `MODEL = "cyttic/trocr-hebrew-synthetic-cont-unfrozen"`.
