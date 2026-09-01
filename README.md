# Transformer from Scratch — English → French Translation

A Transformer built from first principles in TensorFlow/Keras — no pretrained models, no `transformers` library — implementing every core component described in Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (NeurIPS, 2017): sinusoidal positional encoding, scaled dot-product attention, multi-head attention, padding/look-ahead masking, and a stacked encoder-decoder architecture. Trained to translate English sentences to French on the `small_vocab` dataset.

This project doubles as a controlled experiment: three versions of the same model, trained under different conditions, compared on held-out test data to see how much training methodology (not architecture) moves the needle.

## Architecture

Every layer is implemented from scratch on top of raw `tf.keras.layers.Dense` / `Embedding` primitives:

- `positional_encoding()` / `get_angles()` — sinusoidal position signal, added to token embeddings
- `create_padding_mask()` / `create_look_ahead_mask()` / `create_masks()` — masking so attention ignores padding and the decoder can't see future tokens
- `scaled_dot_product_attention()` — the `softmax(QKᵀ/√dₖ + mask)V` core
- `MultiHeadAttention` — splits Q/K/V across attention heads, runs attention per head, recombines
- `FeedForward()` — the position-wise expand → ReLU → compress sublayer
- `EncoderLayer` / `Encoder` and `DecoderLayer` / `Decoder` — stacked self-attention + cross-attention + feed-forward blocks with residual connections and layer normalization
- `Transformer` — ties the encoder and decoder together with a final softmax projection to vocabulary space
- `CustomSchedule` — the paper's warmup + inverse-square-root learning rate schedule

**Object-oriented design:** every stateful component (`MultiHeadAttention`, `EncoderLayer`, `Encoder`, `DecoderLayer`, `Decoder`, `Transformer`, `CustomSchedule`) is a class subclassing `tf.keras.layers.Layer`, `tf.keras.Model`, or `tf.keras.optimizers.schedules.LearningRateSchedule`. Each `__init__` builds and stores its trainable weights once; each `call()` (or `__call__()`) reuses those same weights on every forward pass — which is what lets the weights actually accumulate improvements across thousands of training steps, instead of being reinitialized randomly on every call.

## The Experiment: Three Models, Three Training Regimes

Same architecture, same 24-token max length, same tokenizer/vocabulary — the only things that differ are the train/val/test split and how training was allowed to stop.

| | Transformer I | Transformer II | Transformer III |
|---|---|---|---|
| Split | 100% train, no held-out val | 85 / 10 / 5 (train/val/test) | 85 / 10 / 5 (train/val/test) |
| Epoch ceiling | 10 (fixed) | 20 (fixed ceiling) | 100 (ceiling, early stopping enabled) |
| Early stopping | None | `EarlyStopping(patience=5)`, never triggered | `EarlyStopping(patience=5)`, triggered at epoch 30 |
| Train accuracy | 98.76% | 98.82% | 98.89% |
| Val accuracy | — (no val split) | 98.85% | 98.90% |
| **Test accuracy** | **98.77%** | **98.85%** | **98.87%** |

Full per-epoch histories are in `transformer1_history.json`, `transformer2_history.json`, `transformer3_history.json`. `main.py` regenerates `model_comparison.png` (loss/accuracy curves) and `model_comparison_table.png` (this table) from those files.

### Why early stopping mattered more than the numbers suggest

The headline gain from Transformer II → III looks small — about 0.1–0.2 percentage points of test accuracy. On its own, that's an easy number to dismiss. But the *reason* it happened is the actual point: Transformer II was capped at 20 epochs and never came close to triggering early stopping (`stopped_epoch: 0`) — it simply ran out of budget before validation accuracy had a chance to plateau. Transformer III raised the ceiling to 100 epochs with the exact same `patience=5` stopping rule, and early stopping *did* trigger, at epoch 30 — meaning the model kept genuinely improving past epoch 20, and `restore_best_weights=True` rolled it back to the single best-validation-accuracy checkpoint rather than whatever the last epoch happened to land on.

That's the real value of early stopping: it's not a trick for squeezing out an extra tenth of a percent, it's a guard against overfitting — training only as long as the model is generalizing better, and stopping (with the best weights restored) the moment it stops. A 0.1% gap can look trivial on a small toy dataset like this one, but the same principle scales directly to higher-stakes settings — imbalanced classification, recall-critical tasks scored on F2, medical or safety-relevant models — where a fraction of a percent of accuracy or recall is exactly the margin that matters, and where an overfit model that looks great on training data but generalizes poorly is a real, costly failure mode. Getting early stopping right here, on a small controlled experiment, is practice for getting it right where it counts.

## ⚠️ History Note: How Transformer III Was Actually Made

Transformer II and Transformer III were originally produced from the *same* notebook (`transformer_ii.ipynb`), by editing it in place — there was no separate `transformer_iii.ipynb` file at first. The repo has since been split back apart (`transformer_ii.ipynb` restored to the original Transformer II config, `transformer_iii.ipynb` added as its own file), but the exact diff between the two configs is worth documenting, since it's the entire point of the experiment:

1. **Epoch ceiling** — `EPOCHS = 20` (II) → `EPOCHS = 100` (III). (`EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True)` was already present in both — this was not new to III, it just never got the chance to trigger under II's low ceiling.)
2. **Weight output filename** — `"transformer2_weights.weights.h5"` (II) → `"transformer3_weights.weights.h5"` (III).
3. **History output filename** — `"transformer2_history.json"` (II) → `"transformer3_history.json"` (III).

Everything else — the 85/10/5 split, `embedding_dim=256`, `num_heads=8`, `num_layers=4`, `dropout_rate=0.1`, `batch_size=64`, the model architecture itself — is identical between II and III. `transformer_ii.ipynb` and `transformer_iii.ipynb` now each carry the correct `EPOCHS` value and output filenames for their own model, so running either notebook top-to-bottom reproduces that specific model directly.

A couple of the notebooks' own markdown headers (e.g. "Transformer II — 85/10/5 split...") still say "Transformer II" in both files, a holdover from when they were one file — don't rely on the notebook's in-line prose to tell you which config is active, check the `EPOCHS` value and output filenames directly.

## Environment & GPU Notes

**TensorFlow does not support GPU acceleration on native Windows** (dropped after TF 2.11 — `tf.config.list_physical_devices('GPU')` will always return `[]` there, regardless of your hardware or drivers). **Recommendation: train inside a Linux environment.** This project was trained via **WSL2 (Ubuntu)** with a Python venv running `tensorflow[and-cuda]`, which detected an NVIDIA GPU correctly.

One gotcha if you set this up yourself: even with `tensorflow[and-cuda]` installed via pip, TensorFlow may still fail to find the CUDA/cuDNN shared libraries (`Cannot dlopen some GPU libraries`) unless `LD_LIBRARY_PATH` explicitly includes the `nvidia/*/lib` subdirectories that pip installed inside your venv's `site-packages`. Point `LD_LIBRARY_PATH` at those directories before launching Python/Jupyter if `list_physical_devices('GPU')` comes back empty despite `nvidia-smi` showing your GPU.

## Repo Structure

```
Data/                          small_vocab_en.csv / small_vocab_fr.csv (source sentence pairs)
Notes/                          Personal working notes written while building this
transformer_i.ipynb            Transformer I — full-data training, no validation split
transformer_ii.ipynb           Transformer II — 85/10/5 split, 20-epoch ceiling
transformer_iii.ipynb          Transformer III — 85/10/5 split, 100-epoch ceiling, early stopping triggered
main.py                        Loads all three history JSONs, produces the comparison chart + table
transformer*_history.json      Per-epoch loss/accuracy/val/test results for each model
model_comparison.png           Loss + accuracy curves, all three models
model_comparison_table.png     Summary table (the table above, as an image)
```

Trained weight files (`*.weights.h5`) are gitignored — regenerate them by re-running the notebooks.

## Personal Note

This project was built to genuinely understand the Transformer architecture from the ground up, rather than treating it as a black box import. I plan to carry this understanding of attention, masking, and encoder-decoder architectures forward into my NLP sentiment analysis research with Dr. Kwak.
