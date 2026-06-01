# -*- coding: utf-8 -*-
"""
Fine-tune the Hebrew TrOCR model (ViT encoder + DictaBERT decoder) on the
handwritten dataset.

Pipeline:
  - images come pre-resized to 64px height in the parquet; HebrewBlockProcessor
    mirrors (RTL->LTR), keeps height 64, and tiles them into the 384x384 ViT
    container at load time.
  - labels are tokenised with the DictaBERT tokenizer; pad tokens -> -100.
  - eval uses greedy decoding during training (fast); a final beam-search eval
    is run once at the end for the "real" CER.

Target hardware: single L4 (24 GB, bf16). For a Turing card (e.g. RTX 2080)
use  --precision fp16  since bf16 is unsupported there.

Deps (on the GPU box):
    pip install torch transformers datasets accelerate jiwer pillow

Usage:
    python train.py                         # full run, defaults below
    python train.py --encoder-frozen        # faster first experiment
    python train.py --epochs 1 --max-steps 50   # smoke test
"""

import os
import argparse

import torch
import numpy as np
import jiwer
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from block_processor import HebrewBlockProcessor


def build_collator(processor, tokenizer, max_target_length):
    def collate(batch):
        images = [ex["image"].convert("RGB") for ex in batch]
        texts  = [ex["text"] for ex in batch]

        pixel_values = processor(images)["pixel_values"]

        labels = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        ).input_ids
        # ignore pad tokens in the loss
        labels[labels == tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

    return collate


def build_metrics(tokenizer):
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        # generated preds AND labels can hold -100 padding -> invalid for decode
        pred_ids  = np.where(pred_ids  < 0, tokenizer.pad_token_id, pred_ids)
        label_ids = np.where(label_ids < 0, tokenizer.pad_token_id, label_ids)

        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {
            "cer": jiwer.cer(label_str, pred_str),
            "wer": jiwer.wer(label_str, pred_str),
        }

    return compute_metrics


def main(args):
    # ── model + tokenizer ─────────────────────────────────────────────────────
    print(f"Loading model from {args.model} ...")
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    processor = HebrewBlockProcessor()

    # generation_config takes priority over model.config in recent transformers,
    # so the special tokens must be set on it explicitly or generate() fails.
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id
    model.generation_config.max_new_tokens = None   # use max_length; silences the warning

    if args.encoder_frozen:
        for p in model.encoder.parameters():
            p.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Encoder frozen. Trainable params: {n_train/1e6:.1f}M")

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"Loading dataset {args.dataset} ...")
    ds = load_dataset(args.dataset)
    print(ds)

    collate = build_collator(processor, tokenizer, args.max_target_length)

    # ── training args ─────────────────────────────────────────────────────────
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,                 # -1 = ignore
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=1,                   # greedy during training (fast)
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,   # keep image/text for the custom collator
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collate,
        compute_metrics=build_metrics(tokenizer),
    )

    # ── train ─────────────────────────────────────────────────────────────────
    trainer.train()

    # ── final beam-search eval (the "real" number) ────────────────────────────
    print("\nFinal evaluation with beam search (num_beams=4) ...")
    metrics = trainer.evaluate(
        num_beams=4,
        max_length=args.max_target_length,
        metric_key_prefix="final",
    )
    print(f"  FINAL CER: {metrics['final_cer']:.4f}")
    print(f"  FINAL WER: {metrics['final_wer']:.4f}")

    trainer.save_model(os.path.join(args.output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
    print(f"\nSaved best model -> {os.path.join(args.output_dir, 'best')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="cyttic/trocr-hebrew-untrained")
    p.add_argument("--dataset",     default="cyttic/trocr-hebrew-human")
    p.add_argument("--output-dir",  default="output/trocr-hebrew-human")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--max-steps",   type=int,   default=-1)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--grad-accum",  type=int,   default=1)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--precision",   choices=["bf16", "fp16", "no"], default="bf16")
    p.add_argument("--encoder-frozen", action="store_true")
    main(p.parse_args())
