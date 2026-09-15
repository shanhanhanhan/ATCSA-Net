#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UCR official-split check for ACGC (FordA, Wafer, ElectricDevices).

Called from train_classification_model.py via --ucr NAME, or run directly.
Downloads .ts files into ./ucr_data if they are missing.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split

from baseline_models import ATCSA1D, build_ucr_model, count_params
from train_classification_model import (
    DEVICE,
    LEARNING_RATE,
    SEED,
    compute_metrics,
    save_checkpoint,
)


UCR_DIR = Path('./ucr_data')
UCR_URLS = [
    'https://www.timeseriesclassification.com/Downloads/{name}.zip',
    'https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip',
    'https://timeseriesclassification.com/Downloads/{name}.zip',
]


def _http_get(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (research; ATCSA-Net UCR fetch)'})
    with urlopen(req, timeout=180) as resp:
        return resp.read()
UCR_DATASETS = {
    'FordA': {'expected_len': 500},
    'Wafer': {'expected_len': 152},
    'ElectricDevices': {'expected_len': 96},
}


class SeriesSet(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float().unsqueeze(1)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def parse_ts(text):
    rows, labels = [], []
    data = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith('@data'):
            data = True
            continue
        if not data or line.startswith('@'):
            continue
        if ':' in line:
            body, lab = line.rsplit(':', 1)
            vals = [float(v) for v in body.split(',') if v not in ('', '?')]
            labels.append(lab.strip())
        else:
            parts = [p.strip() for p in line.split(',') if p.strip() != '']
            vals = [float(v) for v in parts[:-1]]
            labels.append(parts[-1])
        rows.append(vals)
    uniq = sorted(set(labels), key=lambda s: (s[0] not in '-0123456789', s))
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y = np.array([mapping[l] for l in labels], dtype=np.int64)
    width = max(len(r) for r in rows)
    x = np.zeros((len(rows), width), dtype=np.float32)
    for i, r in enumerate(rows):
        x[i, :len(r)] = np.asarray(r, dtype=np.float32)
    return x, y, len(uniq)


def _extract_pair(dest, name):
    for suffix in ('.ts', '.tsv', '.txt'):
        train = list(dest.rglob(f'{name}_TRAIN{suffix}'))
        test = list(dest.rglob(f'{name}_TEST{suffix}'))
        if train and test:
            return train[0], test[0]
    return None, None


def download_ucr(name):
    dest = UCR_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    train, test = _extract_pair(dest, name)
    if train and test:
        return train, test

    tsv_mirrors = [
        f'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/{name}/{name}_{{split}}.tsv',
    ]
    got = True
    for split in ('TRAIN', 'TEST'):
        local = dest / f'{name}_{split}.tsv'
        if local.exists():
            continue
        ok = False
        for tmpl in tsv_mirrors:
            url = tmpl.format(split=split)
            print(f'Downloading {url} ...')
            try:
                local.write_bytes(_http_get(url))
                ok = True
                break
            except Exception as exc:
                print(f'  failed: {exc}')
        got = got and ok
    train, test = _extract_pair(dest, name)
    if train and test:
        return train, test

    archive = UCR_DIR / 'UCRArchive2018.zip'
    if not archive.exists():
        zurl = 'https://zenodo.org/records/11198697/files/UCR%20Archive%202018.zip?download=1'
        print(f'Downloading {zurl} (full UCR 2018, ~300MB) ...')
        archive.write_bytes(_http_get(zurl))
    print(f'Extracting {name} from {archive} ...')
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if f'/{name}/' in m.replace('\\', '/') or m.rstrip('/').endswith(f'/{name}')]
        if not members:
            members = [m for m in zf.namelist() if name in m]
        for m in members:
            zf.extract(m, dest)
    train, test = _extract_pair(dest, name)
    if not train:
        raise FileNotFoundError(f'no TRAIN file for {name} after all mirrors')
    return train, test


def parse_tsv(text):
    """UCR 2018 TSV: first column is the label, remaining columns are the series."""
    rows, labels = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.replace(',', '\t').split()
        if len(parts) < 2:
            parts = line.split(',')
        labels.append(parts[0])
        rows.append([float(v) for v in parts[1:]])
    uniq = sorted(set(labels), key=lambda s: (s[0] not in '-0123456789', float(s) if s.replace('-', '', 1).replace('.', '', 1).isdigit() else s))
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y = np.array([mapping[l] for l in labels], dtype=np.int64)
    width = max(len(r) for r in rows)
    x = np.zeros((len(rows), width), dtype=np.float32)
    for i, r in enumerate(rows):
        x[i, :len(r)] = np.asarray(r, dtype=np.float32)
    return x, y, len(uniq)


def load_table(path):
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    if path.suffix.lower() == '.ts' or '@data' in text.lower():
        return parse_ts(text)
    return parse_tsv(text)


def load_ucr(name):
    train_ts, test_ts = download_ucr(name)
    x_tr, y_tr, k1 = load_table(train_ts)
    x_te, y_te, k2 = load_table(test_ts)
    # z-norm per series
    def znorm(x):
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        sd[sd < 1e-6] = 1.0
        return (x - mu) / sd
    return znorm(x_tr), y_tr, znorm(x_te), y_te, max(k1, k2)


def make_loader(ds, shuffle, batch_size):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_flat(model, train_loader, val_loader, epochs, save_path, title):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    ce = nn.CrossEntropyLoss()
    best = -1.0
    for epoch in range(epochs):
        model.train()
        correct = n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            n += yb.size(0)
        sched.step()
        val_acc, _, _ = eval_acc(model, val_loader)
        print(f'  {title} {epoch+1}/{epochs} train {100*correct/max(n,1):.2f}%  val {val_acc:.2f}%')
        if val_acc >= best:
            best = val_acc
            save_checkpoint(save_path, model, extra={'best_val_acc': best})
    return best


@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        pred = model(xb).argmax(1).cpu().numpy()
        preds.extend(pred.tolist())
        labels.extend(yb.numpy().tolist())
    acc = 100.0 * accuracy_score(labels, preds)
    return acc, preds, labels


def symmetric_scores(cm):
    cm = np.asarray(cm, dtype=np.float64)
    rs = cm.sum(1, keepdims=True)
    rs[rs == 0] = 1
    p = cm / rs
    k = p.shape[0]
    s = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                s[i, j] = 0.5 * (p[i, j] + p[j, i])
    return s


def connected_components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a
    groups = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return [sorted(v) for v in groups.values()]


def induce_groups(cm, tau=0.08):
    """Coarse groups from validation confusion. Binary data stays one group."""
    k = cm.shape[0]
    if k <= 2:
        return [list(range(k))]
    s = symmetric_scores(cm)
    edges = [(i, j) for i in range(k) for j in range(i + 1, k) if s[i, j] >= tau]
    comps = connected_components(list(range(k)), edges)
    if len(comps) == 1:
        # fallback: pair the two most confused classes, leave the rest as singletons
        pairs = [((i, j), s[i, j]) for i in range(k) for j in range(i + 1, k)]
        pairs.sort(key=lambda z: -z[1])
        used = set()
        comps = []
        for (i, j), _ in pairs:
            if i in used or j in used:
                continue
            comps.append([i, j])
            used.update((i, j))
            if len(comps) >= 2:
                break
        for i in range(k):
            if i not in used:
                comps.append([i])
    return comps


class GroupMapper:
    def __init__(self, groups):
        self.groups = [tuple(g) for g in groups]
        self.class_to_g = {}
        for gi, g in enumerate(self.groups):
            for c in g:
                self.class_to_g[int(c)] = gi

    def coarse(self, y):
        return self.class_to_g[int(y)]


@torch.no_grad()
def cascade_predict_ucr(coarse, experts, mapper, loader):
    coarse.eval()
    for e in experts:
        if e is not None:
            e.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        g = coarse(xb).argmax(1).cpu().numpy()
        for i, gi in enumerate(g):
            gi = int(gi)
            gt = int(yb[i])
            group = mapper.groups[gi]
            expert = experts[gi]
            if expert is None or len(group) == 1:
                pred = group[0]
            else:
                bit = int(expert(xb[i:i + 1]).argmax(1).item())
                pred = group[min(bit, len(group) - 1)]
            y_true.append(gt)
            y_pred.append(pred)
    return y_true, y_pred


def run_ucr(name, epochs=50, batch_size=32, lr=None):
    if name not in UCR_DATASETS:
        raise ValueError(f'unsupported UCR set {name}, choose {list(UCR_DATASETS)}')
    if lr is not None:
        global LEARNING_RATE
        LEARNING_RATE = lr

    print('=' * 80)
    print(f'UCR {name} on {DEVICE}')
    print('=' * 80)
    x_tr, y_tr, x_te, y_te, n_cls = load_ucr(name)
    in_len = x_tr.shape[1]
    print(f'  train={len(y_tr)} test={len(y_te)} len={in_len} classes={n_cls}')

    # official TRAIN is split 8:2 for validation / early stopping; TEST is untouched
    full_tr = SeriesSet(x_tr, y_tr)
    n_val = max(1, int(0.2 * len(full_tr)))
    n_tr = len(full_tr) - n_val
    train_ds, val_ds = random_split(
        full_tr, [n_tr, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    test_ds = SeriesSet(x_te, y_te)
    train_loader = make_loader(train_ds, True, batch_size)
    val_loader = make_loader(val_ds, False, batch_size)
    test_loader = make_loader(test_ds, False, batch_size)

    ckpt_dir = Path('./checkpoints') / 'ucr' / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results = {'dataset': name, 'n_classes': n_cls, 'length': in_len, 'device': str(DEVICE)}

    for key in ('cnn', 'trans', 'atcsa'):
        print(f'\n--- {name} / {key} ---')
        model = build_ucr_model(key, n_cls, in_len)
        print(f'  params={count_params(model):,}')
        path = ckpt_dir / f'{key}.pth'
        train_flat(model, train_loader, val_loader, epochs, path, f'{name}-{key}')
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)['state_dict'])
        acc, pred, lab = eval_acc(model, test_loader)
        m = compute_metrics(lab, pred, n_cls)
        results[key] = {
            'test_acc': acc,
            'macro_f1': m['macro_f1'],
            'params': count_params(model),
            'confusion_score': m['confusion_score'],
        }
        print(f'  TEST ACC={acc:.2f}%  F1={m["macro_f1"]:.4f}')

    # ACGC on top of the ATCSA-1D trunk: induce groups from the ATCSA val confusion
    print(f'\n--- {name} / ATCSA+ACGC ---')
    atcsa = build_ucr_model('atcsa', n_cls, in_len).to(DEVICE)
    atcsa.load_state_dict(torch.load(ckpt_dir / 'atcsa.pth', map_location=DEVICE, weights_only=False)['state_dict'])
    _, val_pred, val_lab = eval_acc(atcsa, val_loader)
    cm = confusion_matrix(val_lab, val_pred, labels=list(range(n_cls)))
    groups = induce_groups(cm)
    print(f'  induced groups: {groups}')
    mapper = GroupMapper(groups)

    if len(groups) <= 1 or n_cls <= 2:
        # Binary / unsplit trees: ACGC is identical to the flat ATCSA head.
        results['atcsa_acgc'] = {
            **results['atcsa'],
            'groups': groups,
            'note': 'ACGC collapsed to the flat head (K<=2 or a single component).',
        }
        print('  ACGC == flat ATCSA (no extra split)')
    else:
        # remapped loaders
        def remap_coarse(ds):
            xs, ys = [], []
            base = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
            ids = ds.indices if isinstance(ds, torch.utils.data.Subset) else range(len(base))
            for i in ids:
                x, y = base[i]
                xs.append(x)
                ys.append(mapper.coarse(int(y)))
            return SeriesSet(
                torch.stack(xs).squeeze(1).numpy(),
                np.asarray(ys, dtype=np.int64),
            )

        coarse = ATCSA1D(len(groups), in_len).to(DEVICE)
        c_path = ckpt_dir / 'acgc_coarse.pth'
        train_flat(
            coarse,
            make_loader(remap_coarse(train_ds), True, batch_size),
            make_loader(remap_coarse(val_ds), False, batch_size),
            epochs, c_path, f'{name}-coarse',
        )
        coarse.load_state_dict(torch.load(c_path, map_location=DEVICE, weights_only=False)['state_dict'])

        experts = []
        for gi, g in enumerate(groups):
            if len(g) == 1:
                experts.append(None)
                continue
            gset = set(g)
            xs, ys = [], []
            base = train_ds.dataset
            for i in train_ds.indices:
                x, y = base[i]
                yi = int(y)
                if yi in gset:
                    xs.append(x)
                    ys.append(g.index(yi))
            xv, yv = [], []
            for i in val_ds.indices:
                x, y = base[i]
                yi = int(y)
                if yi in gset:
                    xv.append(x)
                    yv.append(g.index(yi))
            if len(set(ys)) < 2:
                experts.append(None)
                continue
            expert = ATCSA1D(len(g), in_len).to(DEVICE)
            e_path = ckpt_dir / f'acgc_e{gi}.pth'
            train_flat(
                expert,
                make_loader(SeriesSet(torch.stack(xs).squeeze(1).numpy(), np.asarray(ys)), True, batch_size),
                make_loader(SeriesSet(torch.stack(xv).squeeze(1).numpy(), np.asarray(yv) if yv else np.asarray(ys[:1])), False, batch_size),
                epochs, e_path, f'{name}-e{gi}',
            )
            expert.load_state_dict(torch.load(e_path, map_location=DEVICE, weights_only=False)['state_dict'])
            experts.append(expert)

        yt, yp = cascade_predict_ucr(coarse, experts, mapper, test_loader)
        m = compute_metrics(yt, yp, n_cls)
        results['atcsa_acgc'] = {
            'test_acc': m['accuracy'] * 100.0,
            'macro_f1': m['macro_f1'],
            'confusion_score': m['confusion_score'],
            'groups': groups,
        }
        print(f'  ATCSA+ACGC TEST ACC={m["accuracy"]*100:.2f}%  F1={m["macro_f1"]:.4f}')

    out = Path('./checkpoints') / f'ucr_{name}.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved {out}')
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ucr', required=True, choices=list(UCR_DATASETS))
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    run_ucr(args.ucr, epochs=args.epochs, batch_size=args.batch_size)
