# English → Persian Translator — a Transformer from scratch

A neural machine translation system built on a hand-written **Encoder–Decoder Transformer** in PyTorch. No pre-trained models and no `nn.Transformer` — every piece (attention, masks, training loop, beam search) is in this repository and commented.

This is the follow-up to the *Simple Transformer* digit-sorting project: the same architecture, scaled up and trained on real parallel text.

```
EN> How are you today?
FA> امروز حالت چطوره؟
```
*(illustrative — actual output depends on how long you train)*

## Pipeline

```
OPUS-100 (en-fa)                                 
      │  prepare_data.py   download · normalize · filter · split · train BPE tokenizers
      ▼
data/  train/valid/test .en/.fa  +  spm_en.model, spm_fa.model
      │  train.py          token-based batching · label smoothing · Noam LR · AMP
      ▼  (model.py)
checkpoints/  best.pt, last.pt
      │  translate.py      beam search · file translation · BLEU / chrF
      ▼
Persian text
```

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads OPUS-100 en-fa (~1M sentence pairs), normalizes Persian text, filters noisy pairs, writes train/valid/test splits, trains two SentencePiece BPE tokenizers. |
| `model.py` | The Transformer: multi-head attention, encoder/decoder layers, masks, greedy decoding, size presets, checkpoint save/load. `python model.py` runs self-tests. |
| `train.py` | Training loop. Auto-configures for GPU/CPU, validates periodically, prints sample translations, saves `best.pt`/`last.pt`, resumable. |
| `translate.py` | Inference: interactive mode, file translation, and BLEU/chrF evaluation on the test set. |

## Requirements

- Python 3.10+
- PyTorch 2.3+ (a CUDA GPU is strongly recommended)
- `pip install datasets sentencepiece sacrebleu`

## Quick start

```bash
pip install torch datasets sentencepiece sacrebleu

python prepare_data.py     # ~5–10 min, once
python model.py            # sanity checks (shapes, causal mask, save/load)
python train.py            # long — see "Hardware" below
python translate.py        # interactive translation
```

### Training options

```bash
python train.py --max-train 200000     # quick experiment on a subset
python train.py --preset base          # Transformer-base size (GPU)
python train.py --max-steps 30000      # stop earlier
python train.py --accum 4              # simulate a 4× larger batch on a small GPU
python train.py --resume               # continue from checkpoints/last.pt
```

`Ctrl+C` stops training safely: the current state is saved to `checkpoints/last.pt`.

### Translation options

```bash
python translate.py                          # interactive, beam search (k=5)
python translate.py --beam 1                 # greedy decoding (faster)
python translate.py --file article.txt       # writes article.fa.txt
python translate.py --bleu --n 1000          # BLEU + chrF on 1000 test sentences
python translate.py --ckpt checkpoints/last.pt
```

## Hardware

`train.py` picks settings automatically:

| Hardware | Preset | Batch (tokens) | Default steps |
|---|---|---|---|
| GPU ≥ 10 GB | `base` — d=512, 8 heads, 6+6 layers, ≈52M params | 8192 | 100k |
| GPU < 10 GB | `small` — d=256, 4 heads, 3+3 layers, ≈10M params | 4096 | 60k |
| CPU | `small` | 2048 | 20k |

CPU training works but is very slow (roughly an hour per 1,000 steps). The free GPU on **Google Colab** is a practical alternative:

```python
# Runtime → Change runtime type → T4 GPU, upload the four .py files, then:
!pip install datasets sentencepiece sacrebleu
!python prepare_data.py
!python train.py --max-steps 30000
```
Colab sessions disconnect after a few hours — rerun with `--resume`. Mount Google Drive to keep `checkpoints/` and `data/` between sessions.

## Design notes

**Data.** OPUS-100 is large but noisy (subtitles, mixed domains). Cleaning removes empty, untranslated, duplicated, over-long (>50 words) and length-mismatched pairs. Persian text is normalized: Arabic `ي ك` → Persian `ی ک`, diacritics and tatweel removed, ZWNJ (half-space) cleaned.

**Tokenization.** Separate 8k-vocabulary SentencePiece BPE models for English and Persian. Subwords mean no word is ever truly unknown. Special IDs: `PAD=0, BOS=1, EOS=2, UNK=3`.

**Model.** Pre-LayerNorm Transformer with sinusoidal positions, `√d_model` embedding scaling, attention via `F.scaled_dot_product_attention` (FlashAttention on supported GPUs) and **weight tying** between the Persian embedding and the output projection.

**Training.**
- Batches are built by token count from length-sorted sentences (little padding, stable memory).
- Cross-entropy with label smoothing 0.1; Adam (β = 0.9, 0.98); Noam schedule with 4,000 warm-up steps; gradient clipping at 1.0.
- Mixed precision (bf16 or fp16) on GPU; optional gradient accumulation.
- Validation loss/perplexity and three sample translations every 1,000 steps; best checkpoint kept by validation loss.

**Decoding.** Beam search (k=5) with the GNMT length penalty `((5+L)/6)^0.6`; batched greedy decoding for fast BLEU evaluation.

**Checkpoints** store the model config alongside the weights, so `translate.py` rebuilds the right architecture automatically.

## Expected results

| Training stage | Typical output |
|---|---|
| ~1k steps | repeated filler words |
| ~5k steps | correct common words, broken grammar |
| 20k+ steps | short, simple sentences mostly understandable |
| full run on GPU | reasonable on everyday sentences, weak on long or technical text |

A BLEU score of roughly **10–20** on OPUS-100 test is a realistic target for a model trained from scratch at this scale. For production-quality translation, fine-tuning a pre-trained multilingual model (e.g. NLLB-200) is the usual route.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: data/spm_en.model` | Run `prepare_data.py` first, from the same folder. |
| `CUDA out of memory` | Lower `--max-tokens` (e.g. 2048) and use `--accum 2` or `--accum 4`. |
| Loss stuck / `nan` | Lower `--lr-factor` (e.g. 0.5); check that `data/` was not regenerated mid-training. |
| Changed tokenizer or `prepare_data.py` settings | Delete `data/*.ids.pkl` (encoded-data cache) and start training from scratch. |
| Repetitive translations | Train longer; use beam search instead of `--beam 1`. |

## Reference

- Vaswani et al., *Attention Is All You Need*, 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Wu et al., *Google's Neural Machine Translation System* (length penalty), 2016 — [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)
- Kudo & Richardson, *SentencePiece*, 2018 — [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
- Zhang et al., *OPUS-100*, 2020 — [arXiv:2004.11867](https://arxiv.org/abs/2004.11867)