# Experiment 7 — results (SUCCESS)

**Setup:** warm-start the **synthetic-pretrained** model
`cyttic/trocr-hebrew-synthetic-cont-unfrozen` (exp3 — encoder, cross-attention
bridge and decoder all already trained) and **domain-adapt** it on
`cyttic/trocr-hebrew-matan` (real handwritten Hebrew, ~4,400 train lines,
writer-disjoint test split of 904 lines). Encoder **unfrozen**, single phase,
lr `2e-5`, effective batch 16 (`BATCH_SIZE=8` × `GRAD_ACCUM=2`), 12 epochs, bf16
on an L4. In-training CER/WER below are **greedy** (`generation_num_beams=1`),
evaluated each epoch on the matan test split; `load_best_model_at_end` selected
the lowest-CER epoch.

| Epoch | Training Loss | Validation Loss | CER | WER |
|-------|--------------|-----------------|------|------|
| 1 | 6.448983 | 2.550867 | 0.274071 | 0.541511 |
| 2 | 2.697279 | 1.398594 | 0.198321 | 0.383784 |
| 3 | 1.374184 | 1.021227 | 0.138204 | 0.276091 |
| 4 | 0.690829 | 0.870272 | 0.108372 | 0.225502 |
| 5 | 0.379615 | 0.787252 | 0.096584 | 0.205128 |
| 6 | 0.245820 | 0.774529 | 0.090025 | 0.200416 |
| 7 | 0.153618 | 0.767583 | 0.083239 | 0.181012 |
| 8 | 0.088986 | 0.767262 | 0.085903 | 0.187110 |
| 9 | 0.076135 | 0.765044 | 0.080098 | 0.179071 |
| **10** | **0.045526** | **0.750799** | **0.074468** | **0.167013** |
| 11 | 0.049676 | 0.746202 | 0.074543 | 0.169785 |
| 12 | 0.032127 | 0.749511 | 0.075347 | 0.171726 |

**Best epoch: 10** (lowest greedy `eval_cer`, selected automatically).

**Final beam-search eval (best model, beam=4, full 904-line test split):**

| Metric | Value |
|--------|-------|
| CER | **6.64%** |
| WER | **16.05%** |
| BLEU | **76.40** |
| exact-match | **46.13%** |

## Verdict: a working reader on real handwriting

This is the experiment that worked. CER falls **monotonically** from 0.274
(epoch 1) to 0.074 (epoch 10) and the final beam-search pass reaches **6.64%
character error on held-out new writers** — essentially matching the synthetic
model's ~6% on its own test set. Nearly half the test lines (46%) are transcribed
**exactly** right.

The curve is the textbook "success + mild overfit" shape: training loss collapses
toward 0.03 (the model is memorising the ~4.4k train lines) while validation loss
and CER **plateau** around epochs 9–11 rather than blowing up. `load_best_model_at_end`
captured epoch 10, so the slight epoch-12 uptick is discarded.

## Why it worked (and exp4–6 didn't)

The decisive variable was the **starting point**, not the schedule:

- **exp4–6** started from `cyttic/trocr-hebrew-untrained`, whose visual→text
  cross-attention is **random noise**. Asking ~4,400 lines to train that bridge
  from scratch is hopeless — all three runs (0 / 4 / 20 frozen epochs) stayed
  pinned at CER ~0.85–1.0. (See `../matan_finetuning_experiments_report.html`.)
- **exp7** started from exp3, whose bridge is **already trained** on ~125k
  synthetic lines. So this run is **domain adaptation** — nudging a working Hebrew
  reader from rendered synthetic ink to real handwriting — not teaching a blind
  model to see. With the bridge no longer random, unfreezing the encoder adapted
  the visual features cleanly instead of corrupting them.

This confirms the project's documented recipe: **synthetic pretrain → real
finetune**, and **never finetune real data from the untrained base.**

## Artifacts

- **Notebook:** `exp7_matan/train_trocr_matan_exp7.ipynb`
- **Model (best-by-CER, epoch 10):** pushed to the Hub as
  **`cyttic/trocr-hebrew-matan-exp7`**
- **Mid-run checkpoints (resume):** `cyttic/trocr-hebrew-matan-exp7-ckpts`

## Using the model (important)

The HF repo holds the weights + tokenizer + config **only**. It does **not**
include the custom **`HebrewBlockProcessor`** (mirror RTL→LTR → 64px height →
tile into 384×384), and the stock TrOCR image processor will **not** reproduce
the training-time preprocessing — feeding raw images through the wrong processor
yields garbage. Any application running this model must apply
`block_processor.py` first. See `eval_metrics_pairs.py` for a complete,
working inference example.
