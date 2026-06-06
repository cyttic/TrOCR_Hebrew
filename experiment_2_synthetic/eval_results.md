# Evaluation results

Log of CER / WER / BLEU measurements for the Hebrew TrOCR models.
Metrics via `eval_metrics.py` (jiwer for CER/WER, sacrebleu for BLEU).

## Model: `models/trocr-hebrew-synthetic-cont`

Stage-1 synthetic pretrain (1 epoch, frozen encoder) **+** continuation
(4 epochs, frozen encoder), warm-started. Evaluated on the **synthetic** test
split `dataset/syntetic_split/test/` (124,996 leak-free samples).

| Date | Samples | Beams | CER | WER | BLEU | Exact-match |
|------|---------|-------|------|------|------|-------------|
| 2026-06-04 | 500   | 1 (greedy) | 0.4301 (43.01%) | 0.8259 (82.59%) | 15.76 | 7.40% |
| 2026-06-04 | 2000  | 4          | **0.4146 (41.46%)** | 0.8084 (80.84%) | 19.02 | 8.50% |

Command (headline run):
```bash
python eval_metrics.py \
  --model models/trocr-hebrew-synthetic-cont \
  --images dataset/syntetic_split/test/images \
  --labels dataset/syntetic_split/test/labels.tsv \
  --beams 4 --limit 2000
```

## Training progression (synthetic CER)

| Stage | Epochs | Encoder | Synthetic CER |
|-------|--------|---------|---------------|
| stage-1 | 1 | frozen | ~0.69 |
| + continuation | 4 | frozen | **0.415** (beam=4, 2k) |

Notes:
- Numbers are consistent with training-time eval (final ~0.45 CER), so no
  train/eval discrepancy or leakage.
- Still **undertrained** — a usable reader wants CER < ~0.10.
- Biggest remaining lever: **unfreeze the encoder** for the next continuation.
