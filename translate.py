import argparse
import sys

import torch

from model import load_checkpoint, PAD_ID, BOS_ID, EOS_ID
from train import load_tokenizers, read_lines, strip_eos, DATA_DIR, SRC, TGT
from prepare_data import NORMALIZE


@torch.no_grad()
def beam_search(model, src_ids, beam=5, alpha=0.6, max_len=None):
    device = next(model.parameters()).device
    src = torch.tensor([src_ids + [EOS_ID]], device=device)
    memory, smask = model.encode(src)
    max_len = max_len or len(src_ids) * 2 + 10
    penalty = lambda L: ((5 + L) / 6) ** alpha

    beams = [([BOS_ID], 0.0)]
    finished = []
    for _ in range(max_len):
        k = len(beams)
        ys = torch.tensor([seq for seq, _ in beams], device=device)
        logits = model.decode(ys, memory.expand(k, -1, -1), smask.expand(k, -1, -1, -1))
        logp = logits[:, -1].float().log_softmax(-1)
        top_lp, top_id = logp.topk(beam, dim=-1)

        candidates = []
        for i, (seq, score) in enumerate(beams):
            for lp, tid in zip(top_lp[i].tolist(), top_id[i].tolist()):
                candidates.append((seq + [tid], score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)

        beams = []
        for seq, score in candidates:
            if seq[-1] == EOS_ID:
                finished.append((seq, score / penalty(len(seq) - 1)))
            else:
                beams.append((seq, score))
            if len(beams) == beam:
                break
        if len(finished) >= beam or not beams:
            break

    if not finished:
        finished = [(seq, score / penalty(len(seq) - 1)) for seq, score in beams]
    best = max(finished, key=lambda f: f[1])[0]
    return strip_eos(best[1:])


@torch.no_grad()
def greedy_batch(model, src_list, batch_size=64):
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(src_list), batch_size):
        chunk = src_list[i: i + batch_size]
        S = max(len(s) for s in chunk) + 1
        src = torch.full((len(chunk), S), PAD_ID, dtype=torch.long)
        for b, s in enumerate(chunk):
            src[b, : len(s) + 1] = torch.tensor(s + [EOS_ID])
        res = model.greedy_decode(src.to(device), max_len=S * 2 + 10)
        outs += [strip_eos(r) for r in res.tolist()]
    return outs


class Translator:
    def __init__(self, ckpt, device=None, beam=5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, info = load_checkpoint(ckpt, self.device)
        self.model.eval()
        self.sp_src, self.sp_tgt = load_tokenizers()
        self.beam = beam
        self.max_src = self.model.cfg.max_len - 2
        print(f"loaded {ckpt}  (step {info.get('step', '?')}, "
              f"{self.model.num_params() / 1e6:.1f}M params, device={self.device})")

    def __call__(self, text):
        ids = self.sp_src.encode(NORMALIZE[SRC](text))[: self.max_src]
        if not ids:
            return ""
        if self.beam > 1:
            out = beam_search(self.model, ids, beam=self.beam)
        else:
            out = greedy_batch(self.model, [ids])[0]
        return self.sp_tgt.decode(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--beam", type=int, default=5, help="1 = greedy")
    p.add_argument("--file", help="per line translation in the file")
    p.add_argument("--bleu", action="store_true", help="BLEU on data/test")
    p.add_argument("--n", type=int, default=1000, help="count of test sentences for BLEU")
    args = p.parse_args()

    tr = Translator(args.ckpt, beam=args.beam)

    if args.bleu:
        import sacrebleu
        srcs = read_lines(f"{DATA_DIR}/test.{SRC}", args.n)
        refs = read_lines(f"{DATA_DIR}/test.{TGT}", args.n)
        print(f"translating {len(srcs)} test sentences (beam={args.beam}) ...")
        if args.beam > 1:
            hyps = []
            for i, s in enumerate(srcs):
                hyps.append(tr(s))
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(srcs)}")
        else:
            ids = [tr.sp_src.encode(s)[: tr.max_src] or [PAD_ID] for s in srcs]
            hyps = [tr.sp_tgt.decode(o) for o in greedy_batch(tr.model, ids)]
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        print(f"\nBLEU = {bleu.score:.2f}    chrF = {chrf.score:.2f}")
        for s, r, h in list(zip(srcs, refs, hyps))[:5]:
            print(f"\nEN : {s}\nREF: {r}\nOUT: {h}")
        return

    if args.file:
        out_path = args.file.rsplit(".", 1)[0] + f".{TGT}.txt"
        lines = read_lines(args.file)
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(tr(line) + "\n")
        print(f"wrote {len(lines)} lines to {out_path}")
        return

    print("Enter the English sentence (for quit enter q): ")
    while True:
        try:
            text = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() == "q":
            break
        if text:
            print("FA>", tr(text))


if __name__ == "__main__":
    main()
