#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External baselines used by the ATCSA trainer: TimeSformer and Mamba-1D.

Both consume the same SCP tensor (B, 1, 128, 128) as ATCSA-Net.
TimeSformer uses divided space-time attention on the waterfall viewed as
T frames. Mamba-1D treats each time row as one token (length 128).
Implementations are from-scratch PyTorch (no pretrained weights).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class _MHSA(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + self.drop(h))


class _MLP(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class DividedSTBlock(nn.Module):
    """TimeSformer-style temporal-then-spatial attention on (B, T, S, C)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.temp = _MHSA(dim, num_heads, dropout)
        self.spat = _MHSA(dim, num_heads, dropout)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        # x: (B, T, S, C)
        b, t, s, c = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(b * s, t, c)
        xt = self.temp(xt).reshape(b, s, t, c).permute(0, 2, 1, 3)
        xs = xt.reshape(b * t, s, c)
        xs = self.spat(xs).reshape(b, t, s, c)
        return self.mlp(xs)


class TimeSformerDAS(nn.Module):
    """Divided space-time transformer on an SCP waterfall.

    (B, 1, 128, 128) is reshaped to T=8 frames of 16 x 128 and patch-embedded
    with 16 x 16 patches (8 spatial tokens per frame).
    """

    def __init__(self, num_classes=5, embed_dim=192, depth=6, num_heads=6,
                 n_frames=8, patch=16, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.use_ccsta = False
        self.n_frames = n_frames
        self.frame_h = 128 // n_frames
        self.patch = patch
        gh, gw = self.frame_h // patch, 128 // patch
        self.n_spatial = gh * gw
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_t = nn.Parameter(torch.zeros(1, n_frames, 1, embed_dim))
        self.pos_s = nn.Parameter(torch.zeros(1, 1, self.n_spatial, embed_dim))
        self.blocks = nn.ModuleList([
            DividedSTBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        nn.init.trunc_normal_(self.pos_s, std=0.02)

    def tokens(self, x):
        b = x.size(0)
        # (B, 1, 128, 128) -> (B*T, 1, fh, 128)
        x = x.view(b, 1, self.n_frames, self.frame_h, 128)
        x = x.reshape(b * self.n_frames, 1, self.frame_h, 128)
        x = self.patch_embed(x)  # (B*T, C, gh, gw)
        x = x.flatten(2).transpose(1, 2)  # (B*T, S, C)
        x = x.view(b, self.n_frames, self.n_spatial, self.embed_dim)
        return x + self.pos_t + self.pos_s

    def forward(self, x, return_features=False):
        tok = self.tokens(x)
        for blk in self.blocks:
            tok = blk(tok)
        pooled = tok.mean(dim=(1, 2))
        feat = self.norm(pooled)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits

    def attention_forward(self, x):
        """Run only the stacked ST blocks (for latency bookkeeping)."""
        tok = self.tokens(x)
        for blk in self.blocks:
            tok = blk(tok)
        return tok


class _MambaBlock(nn.Module):
    """Pure-PyTorch selective SSM (Gu & Dao style) over sequence length L."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def ssm(self, u):
        # u: (B, L, D_inner)
        b, l, d = u.shape
        n = self.d_state
        A = -torch.exp(self.A_log.float())  # (D, N)
        xbc = self.x_proj(u)  # (B, L, 2N+1)
        dt, B, C = xbc.split((1, n, n), dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, D)
        # h: (B, D, N)
        h = u.new_zeros(b, d, n)
        ys = []
        for t in range(l):
            dtt = dt[:, t]  # (B, D)
            # ΔA: (B, D, N)
            decay = torch.exp(dtt.unsqueeze(-1) * A.unsqueeze(0))
            # ΔB u: (B, D, N)
            bu = (dtt.unsqueeze(-1) * B[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1)
            h = h * decay + bu
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + u * self.D
        return y

    def forward(self, x):
        residual = x
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :x.size(1)].transpose(1, 2)
        x_in = F.silu(x_in)
        y = self.ssm(x_in) * F.silu(z)
        return self.norm(residual + self.out_proj(y))


class Mamba1DDAS(nn.Module):
    """1D Mamba encoder on SCP patches: each time row is one token."""

    def __init__(self, num_classes=5, d_model=192, n_layer=6, d_state=16,
                 expand=2, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.use_ccsta = False
        self.input_proj = nn.Linear(128, d_model)
        self.blocks = nn.ModuleList([
            _MambaBlock(d_model, d_state=d_state, expand=expand) for _ in range(n_layer)
        ])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def encode(self, x):
        # x: (B, 1, 128, 128) -> (B, T=128, S=128)
        seq = x.squeeze(1)
        h = self.drop(self.input_proj(seq))
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h.mean(dim=1))

    def forward(self, x, return_features=False):
        feat = self.encode(x)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits

    def attention_forward(self, x):
        """SSM stack only (no classifier), used as the 'attention-stage' timer."""
        seq = self.input_proj(x.squeeze(1))
        for blk in self.blocks:
            seq = blk(seq)
        return seq


class CNN1D(nn.Module):
    """Lightweight 1D CNN for UCR official-split baselines."""

    def __init__(self, num_classes, in_len):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 1, L)
        return self.head(self.features(x).flatten(1))


class Transformer1D(nn.Module):
    """Patch-style 1D transformer for UCR."""

    def __init__(self, num_classes, in_len, d_model=64, n_layer=3, n_head=4, patch=8):
        super().__init__()
        self.num_classes = num_classes
        self.patch = patch
        n_tok = math.ceil(in_len / patch)
        self.n_tok = n_tok
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1, activation='gelu',
        )
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        # x: (B, 1, L)
        b, _, length = x.shape
        pad = (self.patch - length % self.patch) % self.patch
        if pad:
            x = F.pad(x, (0, pad))
        tok = x.view(b, -1, self.patch)
        h = self.embed(tok) + self.pos[:, :tok.size(1)]
        h = self.norm(self.enc(h).mean(dim=1))
        return self.head(h)


class ATCSA1D(nn.Module):
    """1D CNN + chunked temporal attention (UCR stand-in for ATCSA w/o ACGC)."""

    def __init__(self, num_classes, in_len, chunk_size=8, num_heads=4):
        super().__init__()
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.MultiheadAttention(128, num_heads, batch_first=True)
        self.rho = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, num_classes)

    def encode(self, x):
        h = self.backbone(x)  # (B, C, T)
        b, c, t = h.shape
        lc = self.chunk_size
        pad = (lc - t % lc) % lc
        if pad:
            h = F.pad(h, (0, pad))
        t_pad = t + pad
        n_chunks = t_pad // lc
        tok = h.transpose(1, 2).reshape(b * n_chunks, lc, c)
        out, _ = self.attn(tok, tok, tok, need_weights=False)
        out = out.reshape(b, n_chunks * lc, c)[:, :t]
        base = h.transpose(1, 2)[:, :t]
        y = self.norm(base + self.rho * out)
        return y.mean(dim=1)

    def forward(self, x):
        return self.head(self.encode(x))


def build_das_baseline(name, num_classes=5):
    name = name.lower()
    if name in ('timesformer', 'timesformer-das'):
        return TimeSformerDAS(num_classes=num_classes)
    if name in ('mamba', 'mamba-1d', 'mamba1d'):
        return Mamba1DDAS(num_classes=num_classes)
    raise ValueError(f'unknown DAS baseline: {name}')


def build_ucr_model(name, num_classes, in_len):
    name = name.lower()
    if name == 'cnn':
        return CNN1D(num_classes, in_len)
    if name in ('trans', 'transformer'):
        return Transformer1D(num_classes, in_len)
    if name in ('atcsa', 'atcsa_flat'):
        return ATCSA1D(num_classes, in_len)
    raise ValueError(f'unknown UCR model: {name}')
