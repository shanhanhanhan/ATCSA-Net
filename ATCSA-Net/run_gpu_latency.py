#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU latency, memory, and analytical attention FLOPs (batch=1, CUDA sync)."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from atcsa_model import ATCSANet, CCSTA
from train_classification_model import CHECKPOINT_DIR, NUM_HEADS, load_atcsa, load_cascade

DEVICE = torch.device('cuda')
OUT = Path('latency_profile.json')


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def bench_ms(fn, warmup=30, repeats=100):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = torch.tensor(times)
    return {'mean_ms': float(t.mean()), 'std_ms': float(t.std()), 'median_ms': float(t.median())}


def peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    sync()
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def attn_matmul_flops(b, c, t, s, heads, lc):
    """QK^T + AV only. n_chunks = ceil(T/Lc), tokens = Lc*S."""
    n_chunks = (t + lc - 1) // lc
    tokens = lc * s
    d = c // heads
    return 2 * b * heads * n_chunks * tokens * tokens * d


def main():
    print('GPU', torch.cuda.get_device_name(0))
    pre = load_atcsa(CHECKPOINT_DIR / 'atcsa_pretrain.pth', 5, False).to(DEVICE).eval()
    coarse = load_atcsa(CHECKPOINT_DIR / 'atcsa_coarse.pth', 3, False).to(DEVICE).eval()
    s1 = load_atcsa(CHECKPOINT_DIR / 'atcsa_fine_s1.pth', 1, True).to(DEVICE).eval()
    s2 = load_atcsa(CHECKPOINT_DIR / 'atcsa_fine_s2.pth', 1, True).to(DEVICE).eval()
    cas = load_cascade().to(DEVICE).eval()

    x = torch.randn(1, 1, 128, 128, device=DEVICE)

    def feat_of(net):
        with torch.no_grad():
            f = net.conv4(net.conv3(net.conv2(net.conv1(x))))
        return f

    rows = {}
    for name, net, has_attn in (
        ('flat_cnn', pre, False),
        ('coarse', coarse, False),
        ('s1', s1, True),
        ('s2', s2, True),
    ):
        def full(n=net):
            n(x)
        rec = {'full_ms': bench_ms(full), 'mem_mb': peak_mb(full), 'params': sum(p.numel() for p in net.parameters())}
        if has_attn:
            f = feat_of(net)

            def attn(module=net.ccsta, feat=f):
                module(feat)

            rec['attn_ms'] = bench_ms(attn)
            rec['attn_lt_full'] = rec['attn_ms']['mean_ms'] < rec['full_ms']['mean_ms']
        rows[name] = rec

    def three():
        cas.forward_heads(x)

    def typical():
        cas.coarse(x)
        cas.s1(x)

    rows['cascade_three_heads'] = {'full_ms': bench_ms(three), 'mem_mb': peak_mb(three)}
    rows['cascade_typical'] = {'full_ms': bench_ms(typical), 'mem_mb': peak_mb(typical)}

    dummy = ATCSANet(5, True, NUM_HEADS, 2).to(DEVICE).eval()
    with torch.no_grad():
        feat = dummy.conv4(dummy.conv3(dummy.conv2(dummy.conv1(x))))
    b, c, t, s = feat.shape
    lc_rows = []
    for lc in (1, 2, 4, 8, t):
        blk = CCSTA(c, NUM_HEADS, chunk_size=lc).to(DEVICE).eval()
        with torch.no_grad():
            blk.rho.fill_(1.0)

        def fn(module=blk):
            module(feat)

        lc_rows.append({
            'lc': 'full' if lc == t else lc,
            'attn_ms': bench_ms(fn),
            'attn_matmul_flops': attn_matmul_flops(b, c, t, s, NUM_HEADS, lc),
            'qkv_proj_flops': 4 * b * c * c * t * s,  # qkv 3x + proj
        })

    payload = {
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'feat_shape': [b, c, t, s],
        'models': rows,
        'lc': lc_rows,
        'cascade_params': sum(p.numel() for p in cas.parameters()),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({
        'flat_full': rows['flat_cnn']['full_ms']['mean_ms'],
        's1_full': rows['s1']['full_ms']['mean_ms'],
        's1_attn': rows['s1']['attn_ms']['mean_ms'],
        's2_full': rows['s2']['full_ms']['mean_ms'],
        's2_attn': rows['s2']['attn_ms']['mean_ms'],
        'typical': rows['cascade_typical']['full_ms']['mean_ms'],
        'three': rows['cascade_three_heads']['full_ms']['mean_ms'],
        's1_ok': rows['s1']['attn_lt_full'],
        's2_ok': rows['s2']['attn_lt_full'],
        'lc': [(r['lc'], round(r['attn_ms']['mean_ms'], 3), r['attn_matmul_flops']) for r in lc_rows],
    }, indent=2))


if __name__ == '__main__':
    main()
