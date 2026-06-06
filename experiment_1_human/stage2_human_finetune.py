# -*- coding: utf-8 -*-
"""
Stage 2: human finetune, warm-started from the stage-1 synthetic model.

Self-contained (HebrewBlockProcessor is inlined), so you can either:
  - run it directly on the VM:   python stage2_human_finetune.py
  - or open it and copy cells into the notebook with correct indentation.

Edit STAGE1_DIR below if your stage-1 model is somewhere else.
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
)

# ── config ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1_DIR = os.path.join(ROOT, "output/trocr-hebrew-synthetic/final")   # the model stage 1 just trained
DATASET_ID = "cyttic/trocr-hebrew-human"
HUMAN_OUT  = os.path.join(ROOT, "output/trocr-hebrew-human")
EPOCHS     = 4
BATCH_SIZE = 16          # same as stage 1 (fine on the L4, frozen encoder). On a 2080 use 4 + fp16.
PRECISION  = "bf16"      # "bf16" (L4) | "fp16" (RTX 2080) | "no"
MAX_LEN    = 128


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


def main():
    print(f"stage-1 model: {STAGE1_DIR}")
    assert os.path.isdir(STAGE1_DIR), f"stage-1 folder not found: {STAGE1_DIR}"

    # ── data ──────────────────────────────────────────────────────────────────
    ds = load_dataset(DATASET_ID)
    print(ds)

    # ── warm-start from stage 1 ────────────────────────────────────────────────
    model = VisionEncoderDecoderModel.from_pretrained(STAGE1_DIR)
    tok = AutoTokenizer.from_pretrained(STAGE1_DIR)
    proc = HebrewBlockProcessor()

    model.generation_config.decoder_start_token_id = tok.cls_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    model.generation_config.max_new_tokens = None

    for p in model.encoder.parameters():   # keep encoder frozen (same as stage 1)
        p.requires_grad = False

    # ── collator + metrics ─────────────────────────────────────────────────────
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

    # ── train ──────────────────────────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(
        output_dir=HUMAN_OUT,
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
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collate,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    m = trainer.evaluate(num_beams=4, max_length=MAX_LEN, metric_key_prefix="final")
    print(f"\nFINAL human  CER: {m['final_cer']:.4f} | WER: {m['final_wer']:.4f}")

    best = os.path.join(HUMAN_OUT, "best")
    trainer.save_model(best)
    tok.save_pretrained(best)
    print(f"saved -> {best}")
    print("\nNow back it up off the ephemeral VM, e.g.:")
    print(f"  !gsutil cp -r {best} gs://cyttic-trocr-models/human-ft-best")


if __name__ == "__main__":
    main()
