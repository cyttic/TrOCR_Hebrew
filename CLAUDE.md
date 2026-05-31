# CLAUDE.md

Guidance for working in this repo. Hebrew handwriting OCR (HTR) by fine-tuning a
custom TrOCR model.

## What this project is

A Hebrew TrOCR model: the **ViT encoder** from `microsoft/trocr-base-handwritten`
combined with a **DictaBERT** (`dicta-il/dictabert`) decoder. Goal: transcribe
Hebrew handwritten text lines into Unicode text.

The built (untrained) model lives locally in `trocr-hebrew-untrained/` and on the
Hub as **`cyttic/trocr-hebrew-untrained`**.

## Architecture notes (important)

- The encoder→decoder **cross-attention is randomly initialized** — DictaBERT was
  a masked LM, never autoregressive. This mirrors how original TrOCR uses RoBERTa
  as its decoder init. Consequence: the visual→text bridge is effectively trained
  from scratch and is **data-hungry**. DictaBERT is the right decoder choice; the
  bottleneck is data scale, not the decoder.
- Images are NOT fed through the stock TrOCR image processor. A custom
  **`HebrewBlockProcessor`** (`block_processor.py`) does: mirror (RTL→LTR) →
  resize to 64px height → tile left-to-right/top-to-bottom into a 384×384 ViT
  container. This file is custom repo code and is **not** in the HF model repo —
  it must be present wherever the model is trained or run.

## Repo layout / pipeline

| File | Purpose |
|------|---------|
| `build_hebrew_trocr.py` | Builds the encoder+decoder model + processor, saves to `trocr-hebrew-untrained/` |
| `block_processor.py` | `HebrewBlockProcessor` — custom image preprocessing (mirror + tile) |
| `prepare_dataset.py` | SCE dataset (TIF + JSON quads) → `dataset/human/{images,labels.tsv}` |
| `generate_synthetic.py` | Renders ~1M synthetic Hebrew line images → `dataset/syntetic/` (note spelling) |
| `split_dataset.py` | **Document-level** train/test split → `dataset/human_split/{train,test}/` |
| `to_parquet.py` | Split dirs → `dataset/human_parquet/{train,test}-00000-of-00001.parquet` |
| `train.py` | Fine-tune (Seq2SeqTrainer); CLI flags below |
| `train_trocr_hebrew.ipynb` | Interactive/visual version of `train.py` |

## The dataset and its critical caveat

`dataset/human/` = 5,303 line crops from **280 documents** but only **127 unique
transcripts** — it is ~280 writers each copying the same fixed set of ~127 Hebrew
pangram sentences (includes an alphabet line, so all letters are covered).

**Always split by document, never per-line.** A per-line split leaks the same
sentence into train and test; the strong DictaBERT decoder then recites from
memory and CER looks artificially good. `split_dataset.py` handles this
(seed=42, 90/10 by document). Even a clean document split only measures "can it
read a **new writer**?" — NOT "can it read **new words**?", because every document
shares the same 127 sentences. So human-only test CER is optimistic about unseen
vocabulary.

Hosted as HF dataset **`cyttic/trocr-hebrew-human`** (`train`/`test` splits, each
row has `image`, `text`, `source_doc`).

## Training recipe

Correct order is **synthetic pretrain → human finetune**. Doing synthetic *after*
human causes catastrophic forgetting of the human stage. Human-only on 5k makes
the model memorize whole sentences ("sentence recognition") and hallucinate known
sentences on novel text — useful as a pipeline smoke-test/baseline, not a general
reader.

```bash
# smoke test (minutes) — validates the whole pipeline end to end
python train.py --encoder-frozen --epochs 1 --max-steps 50

# first real human-only experiment
python train.py --encoder-frozen
```

`train.py` defaults: `--model cyttic/trocr-hebrew-untrained`,
`--dataset cyttic/trocr-hebrew-human`, bf16, greedy eval per epoch + final
beam-search eval, CER/WER via jiwer on the `test` split, best model →
`output/trocr-hebrew-human/best/`. Key flags: `--encoder-frozen`,
`--precision {bf16,fp16,no}`, `--epochs`, `--batch-size`, `--max-steps`.

## Environment

- **No system Python has the ML deps.** A project `.venv/` (from `/usr/bin/python3`
  3.11, with `datasets`+`pillow`) exists for the data tooling only
  (`to_parquet.py`, `split_dataset.py`). Run those with `.venv/bin/python`.
- **GPUs:** this Debian box has only an **RTX 2080 Super Max-Q, 8 GB** (Turing →
  **bf16 unsupported**, use `--precision fp16` and expect tight memory). Real
  training runs on a **GCP L4 (24 GB, bf16)** — a ~1.5–3h full run for the human
  set. On the L4 install the full stack:
  `pip install torch transformers datasets accelerate jiwer pillow`.
- Download HF data to the VM's local SSD before training (don't stream from GCS
  per-batch); use `dataloader_num_workers>=4`.

## Conventions

- The synthetic output folder is spelled `dataset/syntetic/` (sic). `dataset/` is
  gitignored, as are model weights and `.venv/`.
- Keep `train.py` and the notebook in sync — they implement the same logic.
