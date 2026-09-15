#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATCSA-Net backbone: CNN trunk + Chunk-wise Cascade Spatiotemporal Attention (CCSTA).

CCSTA follows the paper's ANN instantiation of block-wise spatial-temporal
attention in STAtten (arXiv:2409.19764): full spatial mixing inside short
temporal chunks (L_c=2), then a 1x1 projection and a gated residual whose
gate rho is initialized at 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCSTA(nn.Module):
    """Chunk-wise Cascade Spatiotemporal Attention.

    Input feature map is interpreted as (B, C, T, S): T = time, S = fiber axis.
    Temporal axis is split into non-overlapping chunks of length ``chunk_size``.
    Within each chunk, multi-head attention runs over L_c * S tokens so that
    complexity is (T / L_c) * O((L_c * S)^2), matching the paper.
    """

    def __init__(self, channels, num_heads=8, chunk_size=2):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.channels = channels
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # Gated residual, initialized to 0 (attention starts off / warm-up).
        self.rho = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: (B, C, T, S)
        Returns:
            y: (B, C, T, S)
        """
        identity = x
        b, c, t, s = x.shape
        lc = self.chunk_size

        pad_t = (lc - t % lc) % lc
        if pad_t:
            x = F.pad(x, (0, 0, 0, pad_t))
        t_pad = t + pad_t
        n_chunks = t_pad // lc

        qkv = self.qkv(x)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, t_pad, s)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, d, T, S)

        def to_chunk_tokens(tensor):
            # (B, heads, d, n_chunks, Lc, S) -> (B, heads, n_chunks, Lc*S, d)
            tensor = tensor.reshape(b, self.num_heads, self.head_dim, n_chunks, lc, s)
            tensor = tensor.permute(0, 1, 3, 4, 5, 2).contiguous()
            return tensor.reshape(b, self.num_heads, n_chunks, lc * s, self.head_dim)

        q = to_chunk_tokens(q)
        k = to_chunk_tokens(k)
        v = to_chunk_tokens(v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, chunks, Lc*S, d)

        out = out.reshape(b, self.num_heads, n_chunks, lc, s, self.head_dim)
        out = out.permute(0, 1, 5, 2, 3, 4).contiguous()
        out = out.reshape(b, c, t_pad, s)
        if pad_t:
            out = out[:, :, :t, :]

        # Concatenate chunk outputs along time, 1x1 project, gated residual.
        # Identity keeps full temporal resolution for the CNN trunk.
        x_attn = self.proj(out)
        return identity + self.rho * x_attn


class FocalLoss(nn.Module):
    """Multi-class focal loss used on coarse / flat heads."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """Optional branch-local metric regularizer for the hardest pair (S2)."""
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())
    dist = torch.cdist(embeddings, embeddings, p=2)
    labels = labels.view(-1)
    pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    neg_mask = ~pos_mask
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    pos_mask = pos_mask & ~eye

    if pos_mask.any() and neg_mask.any():
        pos_dist = dist.masked_fill(~pos_mask, 0.0).max(dim=1).values
        neg_dist = dist.masked_fill(~neg_mask, 1e6).min(dim=1).values
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if valid.any():
            return F.relu(pos_dist[valid] - neg_dist[valid] + margin).mean()
    return embeddings.new_zeros(())


class ATCSANet(nn.Module):
    """CNN classifier used by cascade nodes.

    The convolutional trunk is a 4-layer CNN. CCSTA is inserted after the
    last convolution when ``use_ccsta=True`` (binary experts). The coarse
    3-way router is trained with ``use_ccsta=False``.
    ``num_classes=1`` produces a single logit for BCE binary heads.
    """

    def __init__(self, num_classes=5, use_ccsta=True, num_heads=8, chunk_size=2, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.use_ccsta = use_ccsta

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.ccsta = CCSTA(256, num_heads=num_heads, chunk_size=chunk_size) if use_ccsta else None
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        # Projection used only when triplet regularization is enabled (S2).
        self.metric_head = nn.Linear(256, 64)

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.ccsta is not None:
            x = self.ccsta(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)

    def forward(self, x, return_features=False):
        feat = self.extract_features(x)
        logits = self.fc(feat)
        if return_features:
            metric_feat = F.normalize(self.metric_head(feat), dim=1)
            return logits, metric_feat
        return logits

    def set_rho_trainable(self, trainable):
        if self.ccsta is None:
            return
        self.ccsta.rho.requires_grad = trainable
        if not trainable:
            with torch.no_grad():
                self.ccsta.rho.zero_()
