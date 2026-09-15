#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train TimeSformer, Mamba-1D, and UCR tables, then profile GPU latency."""

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

PY = sys.executable
ROOT = Path(__file__).resolve().parent


def run(cmd):
    print('\n>>>', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def profile_das():
    from baseline_models import build_das_baseline, count_params
    from train_classification_model import DEVICE, CHECKPOINT_DIR

    def bench(fn, warmup=20, repeats=80):
        for _ in range(warmup):
            fn()
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(sum(times) / len(times))

    x = torch.randn(1, 1, 128, 128, device=DEVICE)
    out = {}
    for name in ('timesformer', 'mamba'):
        model = build_das_baseline(name, 5).to(DEVICE).eval()
        path = CHECKPOINT_DIR / f'{name}_flat.pth'
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        with torch.no_grad():
            full = bench(lambda: model(x))
            attn = bench(lambda: model.attention_forward(x))
        out[name] = {
            'params': count_params(model),
            'full_ms': full,
            'attn_ms': attn,
            'attn_lt_full': attn < full,
        }
        print(name, out[name])
    Path('external_latency.json').write_text(
        json.dumps(out, indent=2), encoding='utf-8',
    )
    return out


def main():
    run([PY, 'train_classification_model.py', '--model', 'timesformer', '--epochs', '50'])
    run([PY, 'train_classification_model.py', '--model', 'mamba', '--epochs', '50'])
    for name in ('FordA', 'Wafer', 'ElectricDevices'):
        run([PY, 'train_classification_model.py', '--ucr', name, '--epochs', '50'])
    print('\nProfiling DAS baseline latency ...')
    profile_das()
    print('All external baselines finished.')


if __name__ == '__main__':
    main()
