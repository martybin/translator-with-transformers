"""
مترجم انگلیسی → فارسی — مرحلهٔ ۳: آموزش
========================================
پیش‌نیاز: اجرای prepare_data.py (پوشهٔ data/) و فایل model.py کنار همین فایل.

اجرا:
    python train.py                         # تنظیمات خودکار بر اساس سخت‌افزار
    python train.py --max-train 200000      # فقط ۲۰۰ هزار جمله (آزمایش سریع)
    python train.py --preset base           # مدل بزرگ (GPU لازم است)
    python train.py --resume                # ادامهٔ آموزش از checkpoints/last.pt

وسط آموزش با Ctrl+C متوقف کنید: آخرین وضعیت ذخیره می‌شود و بعداً با --resume ادامه می‌دهید.
"""
import argparse
import math
import os
import pickle
import random
import time

import torch
import torch.nn.functional as F
import sentencepiece as spm

from model import build_model, save_checkpoint, load_checkpoint, PAD_ID, BOS_ID, EOS_ID

DATA_DIR = "data"
SRC, TGT = "en", "fa"


def load_tokenizers():
    sp_src = spm.SentencePieceProcessor(model_file=f"{DATA_DIR}/spm_{SRC}.model")
    sp_tgt = spm.SentencePieceProcessor(model_file=f"{DATA_DIR}/spm_{TGT}.model")
    return sp_src, sp_tgt


def read_lines(path, limit=None):
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    return lines[:limit] if limit else lines


def encode_split(name, sp_src, sp_tgt, max_len, limit=None):
    cache = f"{DATA_DIR}/{name}.{max_len}.{limit or 'all'}.ids.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    print(f"encoding {name} ...")
    src = sp_src.encode(read_lines(f"{DATA_DIR}/{name}.{SRC}", limit))
    tgt = sp_tgt.encode(read_lines(f"{DATA_DIR}/{name}.{TGT}", limit))
    pairs = [(s[: max_len - 1], t[: max_len - 1]) for s, t in zip(src, tgt) if s and t]
    with open(cache, "wb") as f:
        pickle.dump(pairs, f)
    return pairs


def make_batches(pairs, max_tokens, shuffle=True):
    noise = (lambda: random.random()) if shuffle else (lambda: 0.0)
    order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0]) + noise(), len(pairs[i][1])))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = max(len(pairs[i][0]), len(pairs[i][1])) + 1
        new_max = max(cur_max, L)
        if cur and new_max * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, new_max = [], L
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    if shuffle:
        random.shuffle(batches)
    return batches


def collate(pairs, ids, device):
    S = max(len(pairs[i][0]) for i in ids) + 1
    T = max(len(pairs[i][1]) for i in ids) + 1
    src = torch.full((len(ids), S), PAD_ID, dtype=torch.long)
    tin = torch.full((len(ids), T), PAD_ID, dtype=torch.long)
    tout = torch.full((len(ids), T), PAD_ID, dtype=torch.long)
    for b, i in enumerate(ids):
        s, t = pairs[i]
        src[b, : len(s) + 1] = torch.tensor(s + [EOS_ID])
        tin[b, : len(t) + 1] = torch.tensor([BOS_ID] + t)
        tout[b, : len(t) + 1] = torch.tensor(t + [EOS_ID])
    return src.to(device), tin.to(device), tout.to(device)


def noam_lr(step, d_model, warmup, factor):
    step = max(step, 1)
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def strip_eos(ids):
    out = []
    for t in ids:
        if t == EOS_ID:
            break
        if t != PAD_ID:
            out.append(t)
    return out


@torch.no_grad()
def evaluate(model, valid, device, max_tokens, amp_ctx):
    model.eval()
    total, n_tok = 0.0, 0
    for ids in make_batches(valid, max_tokens, shuffle=False):
        src, tin, tout = collate(valid, ids, device)
        with amp_ctx():
            logits = model(src, tin)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tout.reshape(-1),
                               ignore_index=PAD_ID, reduction="sum")
        total += loss.item()
        n_tok += (tout != PAD_ID).sum().item()
    model.train()
    return total / n_tok


@torch.no_grad()
def show_samples(model, samples, sp_src, sp_tgt, device):
    model.eval()
    for en, fa in samples:
        src = torch.tensor([sp_src.encode(en) + [EOS_ID]], device=device)
        out = strip_eos(model.greedy_decode(src, max_len=src.size(1) * 2 + 10)[0].tolist())
        print(f"   EN : {en}\n   REF: {fa}\n   OUT: {sp_tgt.decode(out)}\n")
    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="auto", choices=["auto", "small", "base"])
    p.add_argument("--max-train", type=int, default=None, help="Only the first N sentences of the training set")
    p.add_argument("--max-tokens", type=int, default=None, help="Token per batch (auto)")
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-len", type=int, default=128, help="Maximum tokens per sentence")
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--lr-factor", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--valid-size", type=int, default=2000)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU: {torch.cuda.get_device_name(0)}  ({gpu_mem:.1f} GB)")
        preset = args.preset if args.preset != "auto" else ("base" if gpu_mem >= 10 else "small")
        max_tokens = args.max_tokens or (8192 if gpu_mem >= 10 else 4096)
        max_steps = args.max_steps or (100_000 if preset == "base" else 60_000)
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print("no GPU found — training on CPU (slow). Consider Google Colab.")
        preset = "small" if args.preset == "auto" else args.preset
        max_tokens = args.max_tokens or 2048
        max_steps = args.max_steps or 20_000

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    amp_ctx = lambda: torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    sp_src, sp_tgt = load_tokenizers()
    train = encode_split("train", sp_src, sp_tgt, args.max_len, args.max_train)
    valid = encode_split("valid", sp_src, sp_tgt, args.max_len)[: args.valid_size]
    samples = list(zip(read_lines(f"{DATA_DIR}/valid.{SRC}", 3), read_lines(f"{DATA_DIR}/valid.{TGT}", 3)))
    print(f"train pairs: {len(train):,}   valid pairs: {len(valid):,}")

    last_path, best_path = f"{args.out}/last.pt", f"{args.out}/best.pt"
    step, best_valid = 0, float("inf")
    if args.resume and os.path.exists(last_path):
        model, ckpt = load_checkpoint(last_path, device)
        step, best_valid = ckpt["step"], ckpt.get("best_valid", float("inf"))
        print(f"resumed from step {step}")
    else:
        model = build_model(preset, src_vocab=sp_src.get_piece_size(),
                            tgt_vocab=sp_tgt.get_piece_size(), dropout=args.dropout,
                            max_len=args.max_len + 2).to(device)
        ckpt = None

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    if ckpt and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    d_model = model.cfg.d_model
    print(f"model: {model.num_params() / 1e6:.1f}M params  d_model={d_model}  "
          f"layers={model.cfg.n_enc_layers}+{model.cfg.n_dec_layers}  "
          f"batch={max_tokens} tokens × accum {args.accum}  max_steps={max_steps}  device={device}"
          f"{'  amp=' + str(amp_dtype).split('.')[-1] if use_amp else ''}")

    def save(path):
        save_checkpoint(path, model, optimizer=opt.state_dict(), step=step, best_valid=best_valid)

    model.train()
    t0, tok_count, loss_sum, loss_n = time.time(), 0, 0.0, 0
    try:
        while step < max_steps:
            batches = make_batches(train, max_tokens)
            for bi in range(0, len(batches) - args.accum + 1, args.accum):
                lr = noam_lr(step + 1, d_model, args.warmup, args.lr_factor)
                for g in opt.param_groups:
                    g["lr"] = lr

                for ids in batches[bi: bi + args.accum]:
                    src, tin, tout = collate(train, ids, device)
                    with amp_ctx():
                        logits = model(src, tin)
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tout.reshape(-1),
                                           ignore_index=PAD_ID, label_smoothing=args.label_smoothing)
                    scaler.scale(loss / args.accum).backward()
                    loss_sum += loss.item()
                    loss_n += 1
                    tok_count += (tout != PAD_ID).sum().item()

                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"step {step:6d} | loss {loss_sum / loss_n:.3f} | lr {lr:.2e} | "
                          f"{tok_count / dt:,.0f} tok/s | {dt / 60:.1f} min")
                    loss_sum, loss_n = 0.0, 0

                if step % args.eval_every == 0:
                    v = evaluate(model, valid, device, max_tokens, amp_ctx)
                    is_best = v < best_valid
                    best_valid = min(best_valid, v)
                    print(f"\n=== step {step} | valid loss {v:.3f} | perplexity {math.exp(v):.1f}"
                          f"{'  ★ best' if is_best else ''} ===")
                    show_samples(model, samples, sp_src, sp_tgt, device)
                    save(last_path)
                    if is_best:
                        save(best_path)

                if step >= max_steps:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted — saving checkpoint ...")

    save(last_path)
    if not os.path.exists(best_path):
        save(best_path)
    print(f"done at step {step}. checkpoints in ./{args.out}/  (best valid loss {best_valid:.3f})")


if __name__ == "__main__":
    main()