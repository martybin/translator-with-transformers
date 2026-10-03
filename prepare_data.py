import os
import re
import random
import unicodedata

from datasets import load_dataset
import sentencepiece as spm

OUT_DIR = "data"
SRC, TGT = "en", "fa"
VOCAB_SIZE = 8000
MAX_WORDS = 50
MAX_TRAIN = None
SEED = 42

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

PERSIAN_CHARS = re.compile(r"[؀-ۿ]")


def normalize_fa(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    s = s.replace("ة", "ه").replace("ـ", "")
    s = re.sub(r"[ً-ْ]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" ?‌ ?", "‌", s)
    return s.strip()


def normalize_en(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


NORMALIZE = {"en": normalize_en, "fa": normalize_fa}


def is_good_pair(src: str, tgt: str) -> bool:
    if not src or not tgt:
        return False
    ns, nt = len(src.split()), len(tgt.split())
    if ns > MAX_WORDS or nt > MAX_WORDS:
        return False
    if max(ns, nt) > 3 * min(ns, nt) + 3:
        return False
    fa_side = src if SRC == "fa" else tgt
    if not PERSIAN_CHARS.search(fa_side):
        return False
    if src.lower() == tgt.lower():
        return False
    return True


def clean_split(split, dedupe=True):
    pairs, seen, dropped = [], set(), 0
    for ex in split:
        src = NORMALIZE[SRC](ex["translation"][SRC])
        tgt = NORMALIZE[TGT](ex["translation"][TGT])
        if not is_good_pair(src, tgt):
            dropped += 1
            continue
        if dedupe:
            key = (src, tgt)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
        pairs.append((src, tgt))
    return pairs, dropped


def save(pairs, name):
    with open(f"{OUT_DIR}/{name}.{SRC}", "w", encoding="utf-8") as fs, \
         open(f"{OUT_DIR}/{name}.{TGT}", "w", encoding="utf-8") as ft:
        for s, t in pairs:
            fs.write(s + "\n")
            ft.write(t + "\n")


def train_tokenizer(lang):
    spm.SentencePieceTrainer.train(
        input=f"{OUT_DIR}/train.{lang}",
        model_prefix=f"{OUT_DIR}/spm_{lang}",
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID, unk_id=UNK_ID,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
    )


if __name__ == "__main__":
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("downloading OPUS-100 en-fa ...")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-fa")
    print({k: len(v) for k, v in ds.items()})

    train, d_tr = clean_split(ds["train"])
    valid, d_va = clean_split(ds["validation"], dedupe=False)
    test, d_te = clean_split(ds["test"], dedupe=False)

    random.shuffle(train)
    if MAX_TRAIN:
        train = train[:MAX_TRAIN]

    print(f"\nafter cleaning:  train={len(train):,} (dropped {d_tr:,})  "
          f"valid={len(valid):,}  test={len(test):,}")

    save(train, "train"); save(valid, "valid"); save(test, "test")

    print("\ntraining SentencePiece tokenizers ...")
    for lang in (SRC, TGT):
        train_tokenizer(lang)

    sp = {l: spm.SentencePieceProcessor(model_file=f"{OUT_DIR}/spm_{l}.model") for l in (SRC, TGT)}
    print("\n=== samples ===")
    for s, t in train[:5]:
        print(f"{SRC}: {s}")
        print(f"{TGT}: {t}")
        print(f"   {SRC} tokens: {sp[SRC].encode(s, out_type=str)}")
        print(f"   {TGT} tokens: {sp[TGT].encode(t, out_type=str)}\n")

    lens = sorted(len(sp[TGT].encode(t)) for _, t in train[:20000])
    print(f"{TGT} token length: median={lens[len(lens)//2]}  "
          f"p95={lens[int(len(lens)*0.95)]}  max={lens[-1]}")
    print("\ndone. files in ./data/")
