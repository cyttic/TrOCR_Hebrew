# =====================================================================
#  Kaggle T4 — full CER/WER/BLEU eval of the continued-synthetic model
#  Paste this into ONE Kaggle cell. Notebook settings: GPU T4 + Internet ON.
#  Both model and test data are pulled from the HuggingFace Hub.
# =====================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "sacrebleu"], check=False)

import time
import torch
import numpy as np
import jiwer
import sacrebleu
from PIL import Image, ImageOps
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, AutoTokenizer

MODEL   = "cyttic/trocr-hebrew-synthetic-cont"   # pushed from the workstation
DATASET = "cyttic/trocr-hebrew-synthetic"        # test split lives here
BEAMS   = 4
BATCH   = 16        # T4 16GB + fp16; drop to 8 if OOM
MAXLEN  = 128
LIMIT   = 0         # 0 = all 124,996 ; set e.g. 2000 for a quick check

device = "cuda"


# ── HebrewBlockProcessor (inlined) ───────────────────────────────────
class HebrewBlockProcessor:
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


# ── model + tokenizer ────────────────────────────────────────────────
model = VisionEncoderDecoderModel.from_pretrained(MODEL).to(device).eval().half()  # fp16 for T4
tok = AutoTokenizer.from_pretrained(MODEL)
proc = HebrewBlockProcessor()
model.generation_config.decoder_start_token_id = tok.cls_token_id
model.generation_config.pad_token_id = tok.pad_token_id
model.generation_config.eos_token_id = tok.sep_token_id
model.generation_config.max_new_tokens = None

# ── data ─────────────────────────────────────────────────────────────
ds = load_dataset(DATASET, split="test")
if LIMIT:
    ds = ds.select(range(LIMIT))
N = len(ds)
print(f"test samples: {N}  | beams={BEAMS} | batch={BATCH}", flush=True)

# ── eval loop (batched) ──────────────────────────────────────────────
refs, hyps = [], []
t0 = time.time()
for start in range(0, N, BATCH):
    batch = ds[start:start + BATCH]                  # dict: {'image':[...], 'text':[...]}
    imgs = [im.convert("RGB") for im in batch["image"]]
    pv = proc(imgs)["pixel_values"].to(device, dtype=model.dtype)
    with torch.no_grad():
        ids = model.generate(pv, num_beams=BEAMS, max_length=MAXLEN)
    hyps.extend(tok.batch_decode(ids, skip_special_tokens=True))
    refs.extend(batch["text"])
    done = min(start + BATCH, N)
    if done % (BATCH * 20) == 0 or done == N:
        el = time.time() - t0
        rate = done / el
        eta = (N - done) / rate if rate else 0
        print(f"  {done}/{N}  | {rate:.1f} img/s | elapsed {el/60:.1f}m | ETA {eta/60:.1f}m", flush=True)

# ── metrics ──────────────────────────────────────────────────────────
cer = jiwer.cer(refs, hyps)
wer = jiwer.wer(refs, hyps)
bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
exact = sum(r.strip() == h.strip() for r, h in zip(refs, hyps)) / len(refs)

print(f"\nDONE in {(time.time()-t0)/60:.1f} min  |  CER {cer*100:.2f}%  WER {wer*100:.2f}%  BLEU {bleu:.2f}")

# ── nice report output ───────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# 1) styled summary table
summary = pd.DataFrame(
    [["Samples", f"{N:,}"], ["Beam size", str(BEAMS)],
     ["CER", f"{cer*100:.2f}%"], ["WER", f"{wer*100:.2f}%"],
     ["BLEU", f"{bleu:.2f}"], ["Exact-match", f"{exact*100:.2f}%"]],
    columns=["Metric", "Value"])
display(
    summary.style.hide(axis="index")
    .set_caption("Hebrew TrOCR — full synthetic test set (beam search)")
    .set_properties(**{"font-size": "15px", "text-align": "left", "padding": "6px 22px"})
    .set_table_styles([
        {"selector": "caption", "props": [("font-size", "17px"), ("font-weight", "bold"),
                                          ("padding", "10px"), ("color", "#222")]},
        {"selector": "th", "props": [("background-color", "#1f77b4"), ("color", "white"),
                                     ("font-size", "15px"), ("text-align", "left"),
                                     ("padding", "6px 22px")]},
        {"selector": "td", "props": [("border-bottom", "1px solid #ddd")]},
    ])
)

# 2) bar chart of error rates
fig, ax = plt.subplots(figsize=(6, 3.6))
bars = ax.bar(["CER", "WER"], [cer * 100, wer * 100], color=["#1f77b4", "#d62728"], width=0.55)
for b, v in zip(bars, [cer * 100, wer * 100]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("error rate (%)")
ax.set_ylim(0, max(cer, wer) * 100 * 1.25)
ax.set_title(f"Full synthetic test  (N={N:,}, beam={BEAMS})")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()

# 3) sample predictions (best + worst) -- Hebrew renders correctly in the HTML table
per_cer = [jiwer.cer(r, h) for r, h in zip(refs, hyps)]
order = sorted(range(N), key=lambda i: per_cer[i])
pick = order[:5] + order[-5:]
ex = pd.DataFrame(
    [[refs[i], hyps[i], f"{per_cer[i]*100:.0f}%", "✓" if refs[i].strip() == hyps[i].strip() else ""]
     for i in pick],
    columns=["Ground truth", "Prediction", "CER", "Match"])
display(
    ex.style.hide(axis="index")
    .set_caption("Sample predictions (5 best + 5 worst)")
    .set_properties(subset=["Ground truth", "Prediction"], **{"text-align": "right", "font-size": "15px"})
    .set_properties(subset=["CER", "Match"], **{"text-align": "center"})
    .set_table_styles([
        {"selector": "caption", "props": [("font-size", "16px"), ("font-weight", "bold"), ("padding", "8px")]},
        {"selector": "th", "props": [("background-color", "#444"), ("color", "white"), ("padding", "5px 14px")]},
    ])
)
