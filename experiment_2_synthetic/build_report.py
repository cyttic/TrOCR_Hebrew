# -*- coding: utf-8 -*-
"""Build a .docx training report: CER/WER progression plot + tables + final eval."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

HTML = "/home/cyttic/Downloads/report 37578f1517ef8005b680f324a7255c4c.html"
COLS = ["step", "train_loss", "val_loss", "cer", "wer"]
PLOT = "training_curve.png"
OUT  = "TrOCR_Hebrew_training_report.docx"
STAGE1_LEN = 31249   # 1 epoch over 500k synthetic at batch 16

# ── parse the two training tables ─────────────────────────────────────────────
tables = pd.read_html(HTML)
stage1 = tables[0].copy(); stage1.columns = COLS
cont   = tables[1].copy(); cont.columns   = COLS
for t in (stage1, cont):
    for c in COLS:
        t[c] = pd.to_numeric(t[c], errors="coerce")

# cumulative step axis: continuation is warm-started after stage 1
stage1["cum"] = stage1["step"]
cont["cum"]   = cont["step"] + STAGE1_LEN
allrows = pd.concat([stage1, cont], ignore_index=True)

# ── plot CER + WER over cumulative steps ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(allrows["cum"], allrows["cer"], marker="o", ms=4, lw=1.6, color="#1f77b4", label="CER")
ax.plot(allrows["cum"], allrows["wer"], marker="s", ms=4, lw=1.6, color="#d62728", label="WER")
ax.axvline(STAGE1_LEN, ls="--", color="gray", lw=1)
ymax = float(allrows[["cer", "wer"]].max().max())
ax.text(STAGE1_LEN * 0.5, ymax * 1.02, "stage-1\n(1 epoch, frozen)", ha="center", va="bottom", fontsize=9, color="gray")
ax.text(STAGE1_LEN + (allrows["cum"].max() - STAGE1_LEN) * 0.5, ymax * 1.02,
        "continuation (4 epochs, frozen)", ha="center", va="bottom", fontsize=9, color="gray")
ax.set_xlabel("cumulative training step (stage-1 + continuation)")
ax.set_ylabel("error rate")
ax.set_title("Hebrew TrOCR — CER / WER over training (synthetic eval subset)")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(0, ymax * 1.12)
fig.tight_layout()
fig.savefig(PLOT, dpi=150)
print("plot ->", PLOT)

# ── build the docx ────────────────────────────────────────────────────────────
doc = Document()
doc.add_heading("Hebrew TrOCR — Training Report", level=0)

doc.add_paragraph(
    "Model: cyttic/trocr-hebrew-untrained (ViT encoder + DictaBERT decoder), "
    "fine-tuned on synthetic Hebrew line images. Two-stage frozen-encoder run: "
    "stage-1 pretrain (1 epoch) followed by a warm-started continuation (4 epochs). "
    "Evaluation on the held-out synthetic test split (leak-free)."
)

doc.add_heading("CER / WER progression", level=1)
doc.add_picture(PLOT, width=Inches(6.2))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
start_cer, end_cer = allrows["cer"].iloc[0], allrows["cer"].iloc[-1]
doc.add_paragraph(
    f"CER falls from {start_cer:.3f} at the start of stage-1 to {end_cer:.3f} at the "
    f"end of the continuation. The curve is still descending — the model is "
    f"learning but undertrained (usable OCR wants CER < ~0.10)."
)

def add_table(title, df):
    doc.add_heading(title, level=2)
    tbl = doc.add_table(rows=1, cols=len(COLS))
    tbl.style = "Light Grid Accent 1"
    hdr = tbl.rows[0].cells
    for i, name in enumerate(["Step", "Train loss", "Val loss", "CER", "WER"]):
        hdr[i].text = name
    for _, r in df.iterrows():
        cells = tbl.add_row().cells
        cells[0].text = f"{int(r['step'])}"
        cells[1].text = f"{r['train_loss']:.4f}"
        cells[2].text = f"{r['val_loss']:.4f}"
        cells[3].text = f"{r['cer']:.4f}"
        cells[4].text = f"{r['wer']:.4f}"

doc.add_heading("Per-step metrics", level=1)
add_table("Stage-1 (1 epoch, frozen encoder)", stage1)
add_table("Continuation (4 epochs, frozen encoder)", cont)

doc.add_heading("Final evaluation (held-out synthetic test)", level=1)
ev = doc.add_table(rows=1, cols=5)
ev.style = "Light Grid Accent 1"
for i, name in enumerate(["Samples", "Beams", "CER", "WER", "BLEU"]):
    ev.rows[0].cells[i].text = name
for row in [("500", "1 (greedy)", "0.4301", "0.8259", "15.76"),
            ("2000", "4", "0.4146", "0.8084", "19.02")]:
    cells = ev.add_row().cells
    for i, v in enumerate(row):
        cells[i].text = v
doc.add_paragraph(
    "Headline number: CER 0.415 (beam=4, 2000 samples). Biggest remaining lever: "
    "unfreeze the encoder for the next continuation run."
)

doc.save(OUT)
print("report ->", OUT)
