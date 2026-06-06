# Training configuration

Key hyperparameters used to fine-tune the Hebrew TrOCR model
(ViT encoder + DictaBERT decoder) on the synthetic line dataset.
Values are from `continue_synthetic_train.py` (the run that produced the
evaluated model). Both stages used the same settings **except epochs**
(stage-1 = 1, continuation = 4).

## Most important parameters

| Parameter | Value | What it does / why it matters |
|-----------|-------|-------------------------------|
| `batch_size` | **16** | Images processed per step before a weight update. Bigger = more stable gradients + faster, but more VRAM. 16 fits the L4 (24 GB) with the encoder frozen. |
| `epochs` | **4** (cont.) / 1 (stage-1) | Full passes over the ~500k synthetic images. More = more learning; the CER curve was still descending at 4 (not converged). |
| `learning_rate` | **5e-5** | Step size for weight updates. Too high → unstable; too low → crawls. Standard TrOCR/transformer fine-tuning value. |
| `encoder_frozen` | **True** | ViT image encoder weights locked; only decoder + cross-attention train. Saves memory/time, but the frozen visual features are the main cap on CER (~0.41). |
| `precision` | **bf16** | 16-bit training → ~2x faster, half the memory, negligible accuracy loss. Works on L4/T4; use **fp16** on the local RTX 2080 (bf16 unsupported there). |
| `warmup_ratio` | **0.1** | LR ramps from ~0 over the first 10% of steps before decaying. Prevents early instability. |
| `weight_decay` | **0.01** | Mild regularization (pulls weights toward 0) → less overfitting. |
| `max_target_length` | **128** | Max tokens per transcript; longer lines are truncated. |

## Supporting settings (sensible defaults, not tuned)

| Parameter | Value | Note |
|-----------|-------|------|
| `gradient_accumulation_steps` | 1 | Effective batch = 16 x 1 = 16. Raise to fake a bigger batch without more VRAM. |
| optimizer | AdamW | Hugging Face default; standard for transformers. |
| LR scheduler | linear (warmup -> linear decay) | LR rises for 10% of steps, then decays to ~0 by the end. |
| eval decoding | greedy (beams=1) during training; **beam=4** for the final number | Greedy is fast for per-step monitoring; beam search gives the better "real" CER. |

## Mental model

- **batch_size + grad_accum** -> how much data per update (stability/speed vs memory).
- **learning_rate + warmup + scheduler** -> how big and how aggressively you step.
- **epochs** -> how long you train.
- **bf16 + encoder_frozen** -> memory/speed levers (frozen encoder is also the current accuracy bottleneck).
- **weight_decay** -> regularization against overfitting.
- **max_target_length + beams** -> the text/decoding side.
