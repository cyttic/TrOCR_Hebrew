# -*- coding: utf-8 -*-
"""
Continue SYNTHETIC pretraining, warm-started from the stage-1 model.

Stage 1 did only 1 epoch on the synthetic set (CER ~0.70 -> undertrained), so this
trains MORE epochs on the SAME synthetic data to build the visual->text bridge
further.

Self-contained (HebrewBlockProcessor inlined). On the Vertex VM:
    python continue_synthetic_train.py

Crash-safety: every checkpoint is mirrored to GCS, and on start the latest
checkpoint is pulled back from GCS and training resumes -- so a lost session
costs at most SAVE_STEPS steps. Set GCS_BACKUP = None to disable.
"""

import os
import numpy as np
import torch
import jiwer
from PIL import Image, ImageOps
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# ── config ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1_DIR  = os.path.join(ROOT, "output/trocr-hebrew-synthetic/final")   # warm-start from here
DATASET_ID  = "cyttic/trocr-hebrew-synthetic"         # SYNTHETIC (continue pretraining)
OUT_DIR     = os.path.join(ROOT, "output/trocr-hebrew-synthetic-cont")
EPOCHS      = 4
BATCH_SIZE  = 16          # same as stage 1 (fine on the L4, frozen encoder)
PRECISION   = "bf16"      # "bf16" (L4) | "fp16" (RTX 2080) | "no"
MAX_LEN     = 128

EVAL_SUBSET = 2000        # eval on a subset; full 125k test per-eval is far too slow
EVAL_STEPS  = 5000        # evaluate every N steps
SAVE_STEPS  = 2000        # checkpoint every N steps
SAVE_LIMIT  = 3           # keep last N local checkpoints

GCS_BACKUP  = "gs://cyttic-trocr-models/synthetic-cont"   # set None to disable


# ── HebrewBlockProcessor (inlined) ────────────────────────────────────────────
class HebrewBlockProcessor:
    """Mirror (RTL->LTR) -> resize to 64px height -> tile into a 384x384 ViT container."""
    TARGET_HEIGHT = 64
    CONTAINER_SIZE = 384
    IMAGE_MEAN = [0.5, 0.5, 0.5]
    IMAGE_STD = [0.5, 0.5, 0.5]

    def __call__(self, images, return_tensors="pt"):
        if not isinstance(images, list):
            images = [images]
        return {"pixel_values": torch.stack([self._process(i) for i in images])}

    def _process(self, image):
        image = image.convert("RGB")
        image = ImageOps.mirror(image)
        w, h = image.size
        new_w = max(1, round(w * self.TARGET_HEIGHT / h))
        image = image.resize((new_w, self.TARGET_HEIGHT), Image.LANCZOS)
        container = Image.new("RGB", (self.CONTAINER_SIZE, self.CONTAINER_SIZE), (255, 255, 255))
        arr = np.array(image)
        src_x, dest_x, dest_y = 0, 0, 0
        while src_x < new_w and dest_y < self.CONTAINER_SIZE:
            chunk_w = min(new_w - src_x, self.CONTAINER_SIZE - dest_x)
            chunk = Image.fromarray(arr[:, src_x:src_x + chunk_w])
            container.paste(chunk, (dest_x, dest_y))
            src_x += chunk_w
            dest_x += chunk_w
            if dest_x >= self.CONTAINER_SIZE:
                dest_x = 0
                dest_y += self.TARGET_HEIGHT
        t = torch.tensor(np.array(container), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor(self.IMAGE_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGE_STD).view(3, 1, 1)
        return (t - mean) / std


# ── GCS backup callback: mirror OUT_DIR to GCS after every checkpoint ──────────
class GCSBackup(TrainerCallback):
    def __init__(self, local_dir, gcs_uri):
        self.local_dir, self.gcs_uri = local_dir, gcs_uri

    def on_save(self, args, state, control, **kwargs):
        # -d makes GCS match local, so save_total_limit deletions propagate
        os.system(f"gsutil -m rsync -r -d {self.local_dir} {self.gcs_uri}")
        print(f"[GCSBackup] mirrored -> {self.gcs_uri}")


def main():
    print(f"warm-start: {STAGE1_DIR}")
    assert os.path.isdir(STAGE1_DIR), f"stage-1 folder not found: {STAGE1_DIR}"

    # pull any existing checkpoints back from GCS so we can resume after a crash
    if GCS_BACKUP:
        os.makedirs(OUT_DIR, exist_ok=True)
        os.system(f"gsutil -m rsync -r {GCS_BACKUP} {OUT_DIR}")

    # ── data ──────────────────────────────────────────────────────────────────
    ds = load_dataset(DATASET_ID)
    print(ds)
    eval_ds = ds["test"].select(range(min(EVAL_SUBSET, len(ds["test"]))))
    print("eval subset:", len(eval_ds))

    # ── warm-start ─────────────────────────────────────────────────────────────
    model = VisionEncoderDecoderModel.from_pretrained(STAGE1_DIR)
    tok = AutoTokenizer.from_pretrained(STAGE1_DIR)
    proc = HebrewBlockProcessor()

    model.generation_config.decoder_start_token_id = tok.cls_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    model.generation_config.max_new_tokens = None

    for p in model.encoder.parameters():   # keep encoder frozen (same as stage 1)
        p.requires_grad = False

    def collate(batch):
        pv = proc([e["image"].convert("RGB") for e in batch])["pixel_values"]
        labels = tok(
            [e["text"] for e in batch],
            padding="longest", truncation=True, max_length=MAX_LEN, return_tensors="pt",
        ).input_ids
        labels[labels == tok.pad_token_id] = -100
        return {"pixel_values": pv, "labels": labels}

    def compute_metrics(pred):
        pi = np.where(pred.predictions < 0, tok.pad_token_id, pred.predictions)
        li = np.where(pred.label_ids   < 0, tok.pad_token_id, pred.label_ids)
        ps = tok.batch_decode(pi, skip_special_tokens=True)
        ls = tok.batch_decode(li, skip_special_tokens=True)
        return {"cer": jiwer.cer(ls, ps), "wer": jiwer.wer(ls, ps)}

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=(PRECISION == "bf16"),
        fp16=(PRECISION == "fp16"),
        predict_with_generate=True,
        generation_max_length=MAX_LEN,
        generation_num_beams=1,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=50,
        save_total_limit=SAVE_LIMIT,
        load_best_model_at_end=False,   # continuation: keep latest (not "best by subset")
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    callbacks = [GCSBackup(OUT_DIR, GCS_BACKUP)] if GCS_BACKUP else []
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=eval_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    last_ckpt = get_last_checkpoint(OUT_DIR) if os.path.isdir(OUT_DIR) else None
    print("Resuming from", last_ckpt) if last_ckpt else print("Training from scratch (warm-started weights)")
    trainer.train(resume_from_checkpoint=last_ckpt)

    m = trainer.evaluate(eval_dataset=eval_ds, num_beams=4, max_length=MAX_LEN, metric_key_prefix="final")
    print(f"\nFINAL synthetic  CER: {m['final_cer']:.4f} | WER: {m['final_wer']:.4f}")

    final = os.path.join(OUT_DIR, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print(f"saved -> {final}")
    if GCS_BACKUP:
        os.system(f"gsutil -m cp -r {final} {GCS_BACKUP}/final")
        print(f"[GCSBackup] final model -> {GCS_BACKUP}/final")


if __name__ == "__main__":
    main()
