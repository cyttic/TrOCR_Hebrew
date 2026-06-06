# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned Hebrew TrOCR model on a test set: CER, WER, BLEU.

Usage:
    python eval_metrics.py --model models/first_iteration \
        --images dataset/human_split/test/images \
        --labels dataset/human_split/test/labels.tsv \
        [--beams 1] [--limit 0]
"""

import os
import argparse

import torch
import jiwer
import sacrebleu
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer

from block_processor import HebrewBlockProcessor


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    proc = HebrewBlockProcessor()
    model.generation_config.decoder_start_token_id = tok.cls_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    model.generation_config.max_new_tokens = None

    rows = []
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                rows.append(parts)
    if args.limit:
        rows = rows[: args.limit]

    refs, hyps = [], []
    bs = max(1, args.batch_size)
    for start in range(0, len(rows), bs):
        chunk = rows[start:start + bs]
        imgs = [Image.open(os.path.join(args.images, f)).convert("RGB") for f, _ in chunk]
        pv = proc(imgs)["pixel_values"].to(device)
        with torch.no_grad():
            ids = model.generate(pv, num_beams=args.beams, max_length=128)
        preds = tok.batch_decode(ids, skip_special_tokens=True)
        for (fname, gt), pred in zip(chunk, preds):
            refs.append(gt)
            hyps.append(pred)
        done = min(start + bs, len(rows))
        if done % (bs * 20) == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}", flush=True)

    cer = jiwer.cer(refs, hyps)
    wer = jiwer.wer(refs, hyps)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    exact = sum(r.strip() == h.strip() for r, h in zip(refs, hyps)) / len(refs)

    print("\n================  RESULTS  ================")
    print(f"  samples     : {len(rows)}  (beams={args.beams})")
    print(f"  CER         : {cer:.4f}   ({cer*100:.2f}%)")
    print(f"  WER         : {wer:.4f}   ({wer*100:.2f}%)")
    print(f"  BLEU        : {bleu:.2f}")
    print(f"  exact-match : {exact*100:.2f}%")
    print("==========================================")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="models/first_iteration")
    p.add_argument("--images", default="dataset/human_split/test/images")
    p.add_argument("--labels", default="dataset/human_split/test/labels.tsv")
    p.add_argument("--beams",  type=int, default=1)
    p.add_argument("--limit",  type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    main(p.parse_args())
