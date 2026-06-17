# Experiment 4 — results

**Setup:** warm-start `cyttic/trocr-hebrew-untrained` (random cross-attention),
finetuned directly on `cyttic/trocr-hebrew-matan` (real handwritten Hebrew,
writer-level split) with the **encoder unfrozen from epoch 1**, 8 epochs,
batch size 16, lr `5e-5`, bf16 on an L4. In-training CER/WER below are
**greedy**-decoded (`generation_num_beams=1`), evaluated each epoch on the
matan test split (904 lines).

| Epoch | Training Loss | Validation Loss | CER | WER |
|-------|--------------|-----------------|------|------|
| 1 | 4.604143 | 3.553077 | 0.875342 | 1.051421 |
| 2 | 1.985469 | 1.954495 | 0.802458 | 1.108940 |
| 3 | 1.532818 | 1.839055 | 0.999799 | 1.269439 |
| 4 | 1.309590 | 1.758803 | 0.932645 | 1.346916 |
| 5 | 1.084191 | 1.685283 | 0.884139 | 1.049203 |
| 6 | 1.015429 | 1.777315 | 0.880620 | 1.239085 |
| 7 | 0.906773 | 1.756004 | 0.855563 | 1.301871 |
| 8 | 0.820280 | 1.778957 | 0.858026 | 1.243936 |

## Verdict: this configuration doesn't converge to a usable reader

Two things stand out, and together they point at the same failure mode:

1. **Training loss keeps falling (4.60 -> 0.82) while validation loss stalls
   and then creeps back up** (bottoms out at epoch 5 ~1.69, ends at epoch 8
   ~1.78). Classic overfit signature — the model is increasingly good at
   memorizing the ~4.4k training lines and *not* learning a visual -> text
   mapping that generalizes to new writers.
2. **CER never drops below ~0.86 and WER stays above 1.0** (WER > 1 means the
   hypothesis needs more word-level edits than the reference even has words —
   i.e. the output is dominated by insertions/garbage, not "close but
   imperfect" transcriptions). There is no real downward trend across all 8
   epochs; it oscillates in a narrow, bad band the whole time.

## Why (the mechanism, as discussed before the run)

`trocr-hebrew-untrained` starts with a **randomly-initialized cross-attention**
— the visual -> text bridge is pure noise on step 1. Unfreezing the ViT encoder
*at the same time* lets noisy gradients from that random cross-attention flow
straight back into the pretrained encoder before cross-attention has learned
anything useful, nudging its good visual features in random directions. With
only ~4.4k training lines (much smaller than the 125k-line synthetic set),
there isn't enough data/steps for the three random/noisy/sensitive components
(cross-attention, now-perturbed encoder, decoder) to jointly recover a stable
representation in 8 epochs — hence loss memorizes the training set while CER/WER
on held-out writers stay essentially flat-bad.

This matches the project's own earlier finding from the synthetic track: the
encoder should **not** be unfrozen while the cross-attention is still
effectively random. exp2 froze the encoder through the whole pretrain so
cross-attention could stabilize first, and only *then* (exp3) unfroze it —
and that ordering is what produced the dramatic CER improvement there.

## Next step

**Experiment 5** (`exp5_matan_staged_unfreeze/`) addresses this directly:
4 epochs with the encoder **frozen** (let cross-attention + decoder learn to
read against stable, known-good visual features first), then 8 more epochs
**unfrozen**. That's the staged schedule that already worked for the synthetic
track, applied here to the matan data.
