import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


@dataclass
class ModelConfig:
    src_vocab: int = 8000
    tgt_vocab: int = 8000
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    n_enc_layers: int = 3
    n_dec_layers: int = 3
    dropout: float = 0.1
    max_len: int = 256
    tie_weights: bool = True


PRESETS = {
    "small": dict(d_model=256, n_heads=4, d_ff=1024, n_enc_layers=3, n_dec_layers=3),
    "base":  dict(d_model=512, n_heads=8, d_ff=2048, n_enc_layers=6, n_dec_layers=6),
}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, x_q, x_kv, mask):
        B, Tq, D = x_q.shape
        q, k, v = self._split(self.w_q(x_q)), self._split(self.w_k(x_kv)), self._split(self.w_v(x_kv))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.w_o(out.transpose(1, 2).reshape(B, Tq, D))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(c.d_model, c.n_heads, c.dropout)
        self.ffn = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.norm1, self.norm2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, src_mask))
        return x + self.drop(self.ffn(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(c.d_model, c.n_heads, c.dropout)
        self.cross_attn = MultiHeadAttention(c.d_model, c.n_heads, c.dropout)
        self.ffn = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.norm1 = nn.LayerNorm(c.d_model)
        self.norm2 = nn.LayerNorm(c.d_model)
        self.norm3 = nn.LayerNorm(c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, tgt_mask))
        x = x + self.drop(self.cross_attn(self.norm2(x), memory, src_mask))
        return x + self.drop(self.ffn(self.norm3(x)))


class Translator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.src_emb = nn.Embedding(cfg.src_vocab, d, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab, d, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d, cfg.max_len)
        self.drop = nn.Dropout(cfg.dropout)
        self.enc_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_enc_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_dec_layers)])
        self.enc_norm, self.dec_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.generator = nn.Linear(d, cfg.tgt_vocab)
        self._init_weights()
        if cfg.tie_weights:
            self.generator.weight = self.tgt_emb.weight

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "emb" not in name:
                nn.init.xavier_uniform_(p)
        for emb in (self.src_emb, self.tgt_emb):
            nn.init.normal_(emb.weight, 0.0, self.cfg.d_model ** -0.5)
            with torch.no_grad():
                emb.weight[PAD_ID].zero_()

    @staticmethod
    def src_mask(src):
        return (src != PAD_ID)[:, None, None, :]

    @staticmethod
    def tgt_mask(tgt):
        T = tgt.size(1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device))
        return (tgt != PAD_ID)[:, None, None, :] & causal

    def _embed(self, emb, tokens):
        return self.drop(self.pos(emb(tokens) * math.sqrt(self.cfg.d_model)))

    def encode(self, src):
        mask = self.src_mask(src)
        x = self._embed(self.src_emb, src)
        for layer in self.enc_layers:
            x = layer(x, mask)
        return self.enc_norm(x), mask

    def decode(self, tgt, memory, src_mask):
        x = self._embed(self.tgt_emb, tgt)
        tmask = self.tgt_mask(tgt)
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, tmask)
        return self.generator(self.dec_norm(x))

    def forward(self, src, tgt_in):
        memory, smask = self.encode(src)
        return self.decode(tgt_in, memory, smask)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=100):
        memory, smask = self.encode(src)
        ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self.decode(ys, memory, smask)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, PAD_ID)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt == EOS_ID
            if done.all():
                break
        return ys[:, 1:]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(preset="small", src_vocab=8000, tgt_vocab=8000, **overrides):
    cfg = ModelConfig(src_vocab=src_vocab, tgt_vocab=tgt_vocab, **{**PRESETS[preset], **overrides})
    return Translator(cfg)


def save_checkpoint(path, model, **extra):
    torch.save({"config": asdict(model.cfg), "model": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = Translator(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


if __name__ == "__main__":
    torch.manual_seed(0)
    for name in PRESETS:
        m = build_model(name)
        print(f"{name:6s}: {m.num_params() / 1e6:5.1f}M parameters")

    m = build_model("small").eval()
    src = torch.tensor([[5, 17, 42, 9, EOS_ID, PAD_ID, PAD_ID],
                        [8, 11, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]])
    tgt = torch.tensor([[BOS_ID, 30, 31, 32], [BOS_ID, 40, PAD_ID, PAD_ID]])
    logits = m(src, tgt)
    print("forward:", tuple(src.shape), "+", tuple(tgt.shape), "->", tuple(logits.shape))

    tgt2 = tgt.clone(); tgt2[0, 3] = 99
    same = torch.allclose(m(src, tgt)[0, :3], m(src, tgt2)[0, :3], atol=1e-5)
    print("causal mask OK:", same)

    save_checkpoint("_test.pt", m)
    m2, _ = load_checkpoint("_test.pt")
    print("save/load OK:", torch.allclose(m2.eval()(src, tgt), logits, atol=1e-6))
    import os; os.remove("_test.pt")

    print("greedy (untrained):", m.greedy_decode(src, max_len=5).tolist())