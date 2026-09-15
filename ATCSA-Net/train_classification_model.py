#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATCSA-Net training: SCP preprocessing + confusion-guided cascade (ACGC).

Reported DAS topology (Table 1; frozen after validation-confusion analysis):
  coarse groups: {0,4} | {2,3} | {1}  (3-class CNN, no CCSTA, cross-entropy)
  fine S2: class 0 vs 4 (CNN + CCSTA, BCE)
  fine S1: class 2 vs 3 (CNN + CCSTA, BCE)
  class 1 is emitted at the coarse stage

The three nodes are trained independently. CCSTA follows the block-wise
spatial-temporal attention in STAtten (arXiv:2409.19764).
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from atcsa_model import ATCSANet, batch_hard_triplet_loss

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Hyper-parameters (paper: Adam, 50 epochs, StepLR, L_c=2)
# ---------------------------------------------------------------------------
BLOCK_WIDTH = 128
BLOCK_HEIGHT = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
RHO_WARMUP_EPOCHS = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TEST_PER_CLASS = 108
N_VAL_PER_CLASS = 108
CHUNK_SIZE = 2
NUM_HEADS = 8
TRIPLET_WEIGHT = 0.0
TRIPLET_MARGIN = 0.3

CATEGORIES = ['人工挖掘', '机械挖掘', '收割机', '旋耕机', '定向钻']
CATEGORIES_EN = [
    'Manual Digging',
    'Mechanical Digging',
    'Harvester',
    'Rotary Tiller',
    'Directional Drill',
]
NUM_CLASSES = len(CATEGORIES)

# Paper hierarchy (must stay identical to the manuscript)
# G0 = {0, 4}, G1 = {2, 3}, G2 = {1}
FINE_S2_CLASSES = (0, 4)  # Manual Digging vs Directional Drill
FINE_S1_CLASSES = (2, 3)  # Harvester vs Rotary Tiller
LEAF_CLASS = 1            # Mechanical Digging

COARSE_NAMES = ['G04 (Manual+Drill)', 'G23 (Harvest+Tiller)', 'G1 (Mech. Dig)']
S2_NAMES = [CATEGORIES_EN[0], CATEGORIES_EN[4]]
S1_NAMES = [CATEGORIES_EN[2], CATEGORIES_EN[3]]

CHECKPOINT_DIR = Path('./checkpoints')
FIGURE_DIR = Path('./figures')


def original_to_coarse(label):
    """Map 5-class label to the 3 coarse superclasses."""
    if label in FINE_S2_CLASSES:
        return 0
    if label in FINE_S1_CLASSES:
        return 1
    return 2


def original_to_binary(label, pair):
    """Map a 5-class label onto {0, 1} for a two-class fine head."""
    if label == pair[0]:
        return 0
    if label == pair[1]:
        return 1
    raise ValueError(f'label {label} is not in pair {pair}')


# ---------------------------------------------------------------------------
# SCP dataset
# ---------------------------------------------------------------------------
def extract_points_from_filename(filename):
    return [int(m) for m in re.findall(r'point_(\d+)', filename)]


def scp_transform(image, point_pos):
    """SCP: 128-wide grid block whose center covers the logged event index."""
    height, width = image.shape
    # Same tiling as the original sliding-window code: non-overlapping 128 blocks.
    start = (int(point_pos) // BLOCK_WIDTH) * BLOCK_WIDTH
    if start + BLOCK_WIDTH > width:
        start = max(0, width - BLOCK_WIDTH)
    crop = image[:, start:start + BLOCK_WIDTH]
    if crop.shape[1] < BLOCK_WIDTH:
        pad = np.zeros((height, BLOCK_WIDTH - crop.shape[1]), dtype=crop.dtype)
        crop = np.concatenate([crop, pad], axis=1)

    crop = cv2.resize(crop, (BLOCK_WIDTH, BLOCK_HEIGHT), interpolation=cv2.INTER_LINEAR)
    crop = crop.astype(np.float32)
    if crop.max() > 1.0:
        crop = crop / 255.0
    return crop


class SCPDataset(Dataset):
    """One standardized 128x128 patch per npy file, centered on the event index."""

    def __init__(self, samples, label_mapper=None):
        """
        Args:
            samples: list of (path, original_label)
            label_mapper: optional fn(original_label) -> training label
        """
        self.patches = []
        self.original_labels = []
        self.paths = []
        self.label_mapper = label_mapper

        for path, orig_label in tqdm(samples, desc='SCP load', leave=False):
            path = Path(path)
            try:
                image = np.load(path)
                points = extract_points_from_filename(path.name)
                point_pos = points[0] if points else image.shape[1] // 2
                patch = scp_transform(image, point_pos)
            except Exception as exc:
                print(f'  skip {path}: {exc}')
                continue
            self.patches.append(patch)
            self.original_labels.append(orig_label)
            self.paths.append(str(path))
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).unsqueeze(0).float()
        orig = self.original_labels[idx]
        label = self.label_mapper(orig) if self.label_mapper else orig
        return patch, torch.tensor(label, dtype=torch.long)


class LabelView(Dataset):
    """Reuse cached SCP patches with a different label map / class filter."""

    def __init__(self, base, mapper=None, allowed=None):
        self.base = base
        self.mapper = mapper
        self.ids = [
            i for i in range(len(base))
            if allowed is None or base.original_labels[i] in allowed
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        patch = torch.from_numpy(self.base.patches[i]).unsqueeze(0).float()
        orig = self.base.original_labels[i]
        label = self.mapper(orig) if self.mapper else orig
        return patch, torch.tensor(label, dtype=torch.long)

    @property
    def original_labels(self):
        return [self.base.original_labels[i] for i in self.ids]


def collect_file_samples(data_path, categories):
    samples = []
    data_path = Path(data_path)
    print('\nIndexing npy files...')
    for class_idx, category in enumerate(categories):
        files = sorted((data_path / category).glob('*.npy'))
        print(f'  {category}: {len(files)} files')
        for f in files:
            samples.append((str(f), class_idx))
    return samples


def equal_per_class_split(samples, n_per_class=540, n_test=N_TEST_PER_CLASS,
                          n_val=N_VAL_PER_CLASS, seed=SEED):
    """Equal 540/class then 108/108/rest, matching the paper test cardinality."""
    by_class = defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)

    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    print('\nPer-class split (cap 540, 108 test / 108 val):')
    for c, name in enumerate(CATEGORIES):
        items = list(by_class[c])
        rng.shuffle(items)
        if len(items) > n_per_class:
            items = items[:n_per_class]
        n = len(items)
        if n >= n_test + n_val + 1:
            n_te, n_va = n_test, n_val
        else:
            n_te = max(1, int(round(n * 0.2)))
            n_va = max(1, int(round(n * 0.2)))
            if n_te + n_va >= n:
                n_te = max(1, n // 5)
                n_va = max(1, n // 5)
        n_tr = n - n_te - n_va
        test.extend(items[:n_te])
        val.extend(items[n_te:n_te + n_va])
        train.extend(items[n_te + n_va:])
        print(f'  {name}: train={n_tr}, val={n_va}, test={n_te} (files {n}/{len(by_class[c])})')
    rng.shuffle(train)
    return train, val, test


def filter_pair(samples, pair):
    return [s for s in samples if s[1] in pair]


# ---------------------------------------------------------------------------
# Metrics / plots
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cs = int(cm.sum() - np.trace(cm))
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        specificities.append(float(tn / (tn + fp)) if (tn + fp) else 0.0)
    return {
        'accuracy': float(acc),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'macro_specificity': float(np.mean(specificities)),
        'kappa': float(kappa),
        'confusion_score': cs,
        'confusion_matrix': cm.tolist(),
        'n_samples': int(len(y_true)),
    }


def plot_confusion(cm, names, title, save_path):
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(np.asarray(cm), annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_curves(history, title, save_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'{title} Accuracy')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_checkpoint(path, model, extra=None):
    payload = {
        'state_dict': model.state_dict(),
        'num_classes': getattr(model, 'num_classes', None),
        'use_ccsta': getattr(model, 'use_ccsta', False),
    }
    if extra:
        payload.update(extra)
    if 'chunk_size' not in payload and getattr(model, 'ccsta', None) is not None:
        payload['chunk_size'] = model.ccsta.chunk_size
    torch.save(payload, path)


def load_atcsa(path, num_classes, use_ccsta=True, chunk_size=None):
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        num_classes = ckpt.get('num_classes', num_classes)
        use_ccsta = ckpt.get('use_ccsta', use_ccsta)
        if chunk_size is None:
            chunk_size = ckpt.get('chunk_size', CHUNK_SIZE)
        state = ckpt['state_dict']
    else:
        state = ckpt
        if chunk_size is None:
            chunk_size = CHUNK_SIZE
    model = ATCSANet(num_classes=num_classes, use_ccsta=use_ccsta,
                      num_heads=NUM_HEADS, chunk_size=chunk_size).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_eval(model, loader, binary=False):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(images)
        if binary:
            logits = logits.view(-1)
            loss = criterion(logits, targets.float())
            pred = (logits > 0).long()
        else:
            loss = criterion(logits, targets)
            pred = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        preds.extend(pred.cpu().numpy().tolist())
        labels.extend(targets.cpu().numpy().tolist())
    n = max(len(labels), 1)
    acc = 100.0 * accuracy_score(labels, preds)
    return total_loss / n, acc, preds, labels


def train_one_model(model, train_loader, val_loader, num_epochs, save_path,
                    title, binary=False, use_triplet=False):
    if binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_acc = -1.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    last_val_preds, last_val_labels = [], []

    print(f'\nTraining {title} on {DEVICE} ...')
    print('=' * 80)
    for epoch in range(num_epochs):
        model.train()
        if RHO_WARMUP_EPOCHS > 0:
            model.set_rho_trainable(epoch >= RHO_WARMUP_EPOCHS)

        running_loss = 0.0
        n_seen = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f'{title} {epoch + 1}/{num_epochs}', leave=False)
        for images, targets in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()

            if use_triplet:
                logits, metric_feat = model(images, return_features=True)
            else:
                logits = model(images)

            if binary:
                logits = logits.view(-1)
                loss = criterion(logits, targets.float())
                pred = (logits > 0).long()
            else:
                loss = criterion(logits, targets)
                pred = logits.argmax(dim=1)

            if use_triplet:
                loss = loss + TRIPLET_WEIGHT * batch_hard_triplet_loss(
                    metric_feat, targets, margin=TRIPLET_MARGIN
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            correct += (pred == targets).sum().item()
            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             acc=f'{100.0 * correct / n_seen:.2f}%')

        train_loss = running_loss / max(n_seen, 1)
        train_acc = 100.0 * correct / max(n_seen, 1)
        val_loss, val_acc, val_preds, val_labels = run_eval(
            model, val_loader, binary=binary
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        last_val_preds, last_val_labels = val_preds, val_labels

        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'train {train_loss:.4f}/{train_acc:.2f}%  '
              f'val {val_loss:.4f}/{val_acc:.2f}%')

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(save_path, model, extra={'best_val_acc': best_val_acc})
            print(f'  [OK] saved {save_path} (val acc {val_acc:.2f}%)')

    print(f'{title} done. best val acc = {best_val_acc:.2f}%')
    return history, last_val_preds, last_val_labels, best_val_acc


def make_loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def make_balanced_loader(dataset):
    labels = dataset.original_labels
    counts = np.bincount(np.asarray(labels), minlength=5).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    sample_w = [float(weights[int(y)]) for y in labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def new_model(num_classes, use_ccsta=True, chunk_size=None):
    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    model = ATCSANet(num_classes=num_classes, use_ccsta=use_ccsta,
                     num_heads=NUM_HEADS, chunk_size=chunk_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params = {n_params:,}  ccsta={use_ccsta}  classes={num_classes}  Lc={chunk_size}')
    return model


# ---------------------------------------------------------------------------
# Cascade inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def cascade_predict(model, loader):
    """Deployed 5-class inference: coarse 3-way, then the matching fine head.

    Coarse argmax chooses one of {0,4} / {2,3} / {1}.
    Group {0,4} goes to S2 (0 vs 4); group {2,3} goes to S1 (2 vs 3);
    group {1} is emitted as Mechanical Digging. This is predicted routing,
    not oracle / in-group assignment by the true label.
    """
    model.eval()
    y_true, y_pred = [], []
    coarse_true, coarse_pred = [], []
    s1_true, s1_pred = [], []
    s2_true, s2_pred = [], []

    for images, orig_labels in loader:
        images = images.to(DEVICE)
        orig = orig_labels.cpu().numpy()
        out = model.forward_heads(images)
        groups = out['coarse'].argmax(dim=1).cpu().numpy()
        s1_logit = out['s1'].cpu().numpy()
        s2_logit = out['s2'].cpu().numpy()

        for i, g in enumerate(groups):
            gt = int(orig[i])
            coarse_true.append(original_to_coarse(gt))
            coarse_pred.append(int(g))

            if g == 0:
                bit = int(s2_logit[i] > 0)
                pred = FINE_S2_CLASSES[bit]
                if gt in FINE_S2_CLASSES:
                    s2_true.append(original_to_binary(gt, FINE_S2_CLASSES))
                    s2_pred.append(bit)
            elif g == 1:
                bit = int(s1_logit[i] > 0)
                pred = FINE_S1_CLASSES[bit]
                if gt in FINE_S1_CLASSES:
                    s1_true.append(original_to_binary(gt, FINE_S1_CLASSES))
                    s1_pred.append(bit)
            else:
                pred = LEAF_CLASS

            y_true.append(gt)
            y_pred.append(int(pred))

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'coarse_true': coarse_true,
        'coarse_pred': coarse_pred,
        's1_true': s1_true,
        's1_pred': s1_pred,
        's2_true': s2_true,
        's2_pred': s2_pred,
    }


@torch.no_grad()
def oracle_ingroup_predict(model, loader):
    """In-group fine classification: route by *true* superclass, then S1/S2.

    Stage-wise S1/S2 protocol: evaluate specialists inside the true group.
    """
    model.eval()
    y_true, y_pred = [], []
    s1_true, s1_pred = [], []
    s2_true, s2_pred = [], []

    for images, orig_labels in loader:
        images = images.to(DEVICE)
        orig = orig_labels.cpu().numpy()
        out = model.forward_heads(images)
        s1_logit = out['s1'].cpu().numpy()
        s2_logit = out['s2'].cpu().numpy()

        for i, gt in enumerate(orig):
            gt = int(gt)
            g = original_to_coarse(gt)
            if g == 0:
                bit = int(s2_logit[i] > 0)
                pred = FINE_S2_CLASSES[bit]
                s2_true.append(original_to_binary(gt, FINE_S2_CLASSES))
                s2_pred.append(bit)
            elif g == 1:
                bit = int(s1_logit[i] > 0)
                pred = FINE_S1_CLASSES[bit]
                s1_true.append(original_to_binary(gt, FINE_S1_CLASSES))
                s1_pred.append(bit)
            else:
                pred = LEAF_CLASS
            y_true.append(gt)
            y_pred.append(int(pred))

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        's1_true': s1_true,
        's1_pred': s1_pred,
        's2_true': s2_true,
        's2_pred': s2_pred,
    }


@torch.no_grad()
def oracle_branch_accuracy(model, loader, pair, head='s1'):
    """Fine-head accuracy on ground-truth members of a pair (paper S1/S2)."""
    model.eval()
    preds, labels = [], []
    for images, orig in loader:
        images = images.to(DEVICE)
        orig_np = orig.cpu().numpy()
        out = model.forward_heads(images)
        logits = out[head].cpu().numpy()
        bits = (logits > 0).astype(int)
        for j, gt in enumerate(orig_np):
            if int(gt) in pair:
                labels.append(original_to_binary(int(gt), pair))
                preds.append(int(bits[j]))
    if not labels:
        return 0.0, [], []
    return 100.0 * accuracy_score(labels, preds), preds, labels


@torch.no_grad()
def eval_cascade_split(model, loader):
    out = cascade_predict(model, loader)
    coarse_acc = 100.0 * accuracy_score(out['coarse_true'], out['coarse_pred'])
    final_acc = 100.0 * accuracy_score(out['y_true'], out['y_pred'])
    return coarse_acc, final_acc, out


class CascadeEnsemble(nn.Module):
    """Three independently trained ATCSA nodes used at inference."""

    def __init__(self, coarse, s1, s2):
        super().__init__()
        self.coarse = coarse
        self.s1 = s1
        self.s2 = s2
        self.use_ccsta = True

    def forward_heads(self, x):
        return {
            'coarse': self.coarse(x),
            's1': self.s1(x).view(-1),
            's2': self.s2(x).view(-1),
        }


def load_cascade(path=None):
    coarse = load_atcsa(CHECKPOINT_DIR / 'atcsa_coarse.pth', num_classes=3, use_ccsta=False)
    s1 = load_atcsa(CHECKPOINT_DIR / 'atcsa_fine_s1.pth', num_classes=1, use_ccsta=True)
    s2 = load_atcsa(CHECKPOINT_DIR / 'atcsa_fine_s2.pth', num_classes=1, use_ccsta=True)
    model = CascadeEnsemble(coarse, s1, s2).to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train ATCSA-Net cascade or external baselines')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument(
        '--model', type=str, default='atcsa',
        choices=['atcsa', 'timesformer', 'mamba'],
        help='atcsa = cascade; timesformer / mamba = flat 5-class SCP baselines',
    )
    parser.add_argument(
        '--ucr', type=str, default='',
        choices=['', 'FordA', 'Wafer', 'ElectricDevices'],
        help='If set, run the official-split UCR protocol instead of DAS',
    )
    return parser.parse_args()


def train_das_baseline(model_name, train_loader, val_loader, test_loader, num_epochs):
    """Flat 5-class TimeSformer / Mamba-1D on the same SCP split as ATCSA."""
    from baseline_models import build_das_baseline, count_params

    print('\n' + '#' * 80)
    print(f'External DAS baseline: {model_name} (flat 5-class, same SCP split)')
    print('#' * 80)
    model = build_das_baseline(model_name, num_classes=5).to(DEVICE)
    print(f'  params = {count_params(model):,}')
    save_path = CHECKPOINT_DIR / f'{model_name}_flat.pth'
    hist, _, _, best = train_one_model(
        model, train_loader, val_loader, num_epochs, save_path,
        title=f'{model_name}-5way', binary=False,
    )
    plot_curves(hist, model_name, FIGURE_DIR / f'{model_name}_curves.png')

    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    _, test_acc, pred, lab = run_eval(model, test_loader, binary=False)
    metrics = compute_metrics(lab, pred, 5)
    print(classification_report(lab, pred, target_names=CATEGORIES, digits=4))
    print(f'{model_name} test ACC={metrics["accuracy"]*100:.2f}%  '
          f'F1={metrics["macro_f1"]:.4f}  CS={metrics["confusion_score"]}')
    out = {
        'model': model_name,
        'params': count_params(model),
        'best_val_acc': best,
        'test': metrics,
    }
    out_path = Path(f'{model_name}_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'Saved {out_path} / {save_path}')
    return out


def main():
    args = parse_args()
    global BATCH_SIZE, LEARNING_RATE
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    num_epochs = args.epochs
    data_path = args.data

    if args.ucr:
        from train_ucr import run_ucr
        run_ucr(args.ucr, epochs=num_epochs, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
        return

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    print(f'Device: {DEVICE}')
    print('Cascade groups: {0,4} | {2,3} | {1}')
    print(f'  S2 fine: {CATEGORIES[0]} vs {CATEGORIES[4]}')
    print(f'  S1 fine: {CATEGORIES[2]} vs {CATEGORIES[3]}')

    split_file = Path('./atcsa_splits.json')
    if split_file.exists() and args.model in ('timesformer', 'mamba'):
        payload = json.loads(split_file.read_text(encoding='utf-8'))
        train_s = [(item['path'], item['label']) for item in payload['train']]
        val_s = [(item['path'], item['label']) for item in payload['val']]
        test_s = [(item['path'], item['label']) for item in payload['test']]
        print(f'Reusing {split_file} (train={len(train_s)} val={len(val_s)} test={len(test_s)})')
    else:
        all_samples = collect_file_samples(data_path, CATEGORIES)
        train_s, val_s, test_s = equal_per_class_split(all_samples)
        split_payload = {
            'train': [{'path': p, 'label': int(l)} for p, l in train_s],
            'val': [{'path': p, 'label': int(l)} for p, l in val_s],
            'test': [{'path': p, 'label': int(l)} for p, l in test_s],
            'categories': CATEGORIES,
            'fine_s2': list(FINE_S2_CLASSES),
            'fine_s1': list(FINE_S1_CLASSES),
            'leaf': LEAF_CLASS,
        }
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_payload, f, ensure_ascii=False, indent=2)
        print('Saved atcsa_splits.json')

    # Shared 5-class tensors (SCP once).
    print('\nBuilding SCP datasets (5-class labels)...')
    train_5 = SCPDataset(train_s)
    val_5 = SCPDataset(val_s)
    test_5 = SCPDataset(test_s)
    train_5_loader = make_loader(train_5, True)
    val_5_loader = make_loader(val_5, False)
    test_5_loader = make_loader(test_5, False)
    print(f'  train={len(train_5)}  val={len(val_5)}  test={len(test_5)}')

    if args.model in ('timesformer', 'mamba'):
        train_das_baseline(
            args.model, train_5_loader, val_5_loader, test_5_loader, num_epochs,
        )
        return

    results = {}

    # ----- Stage 0: flat pre-training (confusion source) -----
    print('\n' + '#' * 80)
    print('Stage 0: 5-class pre-training (flat baseline / confusion matrix)')
    print('#' * 80)
    pre_path = CHECKPOINT_DIR / 'atcsa_pretrain.pth'
    pre_model = new_model(num_classes=5, use_ccsta=False)
    hist, _, _, best = train_one_model(
        pre_model, train_5_loader, val_5_loader, num_epochs, pre_path,
        title='Pretrain-5way', binary=False
    )
    plot_curves(hist, 'Pretrain', FIGURE_DIR / 'pretrain_curves.png')
    pre_model = load_atcsa(pre_path, num_classes=5)
    _, pre_val_acc, pre_val_pred, pre_val_lab = run_eval(pre_model, val_5_loader)
    _, pre_test_acc, pre_test_pred, pre_test_lab = run_eval(pre_model, test_5_loader)
    pre_val_m = compute_metrics(pre_val_lab, pre_val_pred, 5)
    pre_test_m = compute_metrics(pre_test_lab, pre_test_pred, 5)
    plot_confusion(pre_val_m['confusion_matrix'], CATEGORIES_EN,
                   'Pretrain Validation Confusion', FIGURE_DIR / 'pretrain_val_confusion.png')
    plot_confusion(pre_test_m['confusion_matrix'], CATEGORIES_EN,
                   'Pretrain Test Confusion (flat)', FIGURE_DIR / 'pretrain_test_confusion.png')
    print('\nPretrain val confusion (rows=true):')
    print(np.array(pre_val_m['confusion_matrix']))
    print(f'Flat test ACC={pre_test_m["accuracy"] * 100:.2f}%  CS={pre_test_m["confusion_score"]}')
    print('Fixed hierarchy (paper): G0={0,4}, G1={2,3}, G2={1}')
    results['pretrain'] = {
        'best_val_acc': best,
        'val': pre_val_m,
        'test': pre_test_m,
    }

    # ----- Stages 1-2: separate cascade nodes (user/paper topology) -----
    print('\n' + '#' * 80)
    print('Stage 1: coarse 3-way CNN  {0,4} / {2,3} / {1}  (no CCSTA)')
    print('#' * 80)
    train_c = LabelView(train_5, mapper=original_to_coarse)
    val_c = LabelView(val_5, mapper=original_to_coarse)
    coarse_path = CHECKPOINT_DIR / 'atcsa_coarse.pth'
    coarse_model = new_model(num_classes=3, use_ccsta=False)
    hist, _, _, best = train_one_model(
        coarse_model, make_loader(train_c, True), make_loader(val_c, False),
        num_epochs, coarse_path, title='Coarse-3way', binary=False
    )
    plot_curves(hist, 'Coarse', FIGURE_DIR / 'coarse_curves.png')
    results['coarse_best_val_acc'] = best

    print('\n' + '#' * 80)
    print('Stage 2a: fine S1  class 2 vs 3  (CNN+CCSTA)')
    print('#' * 80)
    train_s1 = LabelView(
        train_5, mapper=lambda y: original_to_binary(y, FINE_S1_CLASSES),
        allowed=FINE_S1_CLASSES,
    )
    val_s1 = LabelView(
        val_5, mapper=lambda y: original_to_binary(y, FINE_S1_CLASSES),
        allowed=FINE_S1_CLASSES,
    )
    print(f'  S1 train={len(train_s1)}  val={len(val_s1)}')
    s1_path = CHECKPOINT_DIR / 'atcsa_fine_s1.pth'
    s1_model = new_model(num_classes=1, use_ccsta=True)
    hist, _, _, best = train_one_model(
        s1_model, make_loader(train_s1, True), make_loader(val_s1, False),
        num_epochs, s1_path, title='Fine-S1(2vs3)', binary=True
    )
    plot_curves(hist, 'Fine-S1', FIGURE_DIR / 'fine_s1_curves.png')
    results['s1_best_val_acc'] = best

    print('\n' + '#' * 80)
    print('Stage 2b: fine S2  class 0 vs 4  (CNN+CCSTA + triplet)')
    print('#' * 80)
    train_s2 = LabelView(
        train_5, mapper=lambda y: original_to_binary(y, FINE_S2_CLASSES),
        allowed=FINE_S2_CLASSES,
    )
    val_s2 = LabelView(
        val_5, mapper=lambda y: original_to_binary(y, FINE_S2_CLASSES),
        allowed=FINE_S2_CLASSES,
    )
    print(f'  S2 train={len(train_s2)}  val={len(val_s2)}')
    s2_path = CHECKPOINT_DIR / 'atcsa_fine_s2.pth'
    s2_model = new_model(num_classes=1, use_ccsta=True)
    hist, _, _, best = train_one_model(
        s2_model, make_loader(train_s2, True), make_loader(val_s2, False),
        num_epochs, s2_path, title='Fine-S2(0vs4)', binary=True, use_triplet=True
    )
    plot_curves(hist, 'Fine-S2', FIGURE_DIR / 'fine_s2_curves.png')
    results['s2_best_val_acc'] = best

    # ----- Stage 3: deployed cascade = coarse 3-way then S1/S2 -----
    print('\n' + '#' * 80)
    print('Stage 3: deployed cascade on held-out test set')
    print('  coarse 3-way {0,4} / {2,3} / {1}')
    print('  then S2: 0 vs 4, or S1: 2 vs 3, or emit class 1')
    print('#' * 80)
    cascade_model = load_cascade()

    s1_acc, _, _ = oracle_branch_accuracy(
        cascade_model, test_5_loader, FINE_S1_CLASSES, head='s1'
    )
    s2_acc, _, _ = oracle_branch_accuracy(
        cascade_model, test_5_loader, FINE_S2_CLASSES, head='s2'
    )

    routed = cascade_predict(cascade_model, test_5_loader)
    coarse_m = compute_metrics(routed['coarse_true'], routed['coarse_pred'], 3)
    routed_m = compute_metrics(routed['y_true'], routed['y_pred'], 5)

    out = oracle_ingroup_predict(cascade_model, test_5_loader)
    ingroup_m = compute_metrics(out['y_true'], out['y_pred'], 5)
    s1_m = compute_metrics(out['s1_true'], out['s1_pred'], 2)
    s2_m = compute_metrics(out['s2_true'], out['s2_pred'], 2)

    plot_confusion(coarse_m['confusion_matrix'], COARSE_NAMES,
                   'Coarse Test Confusion', FIGURE_DIR / 'coarse_test_confusion.png')
    plot_confusion(routed_m['confusion_matrix'], CATEGORIES_EN,
                   'Predicted Routing (5-class)',
                   FIGURE_DIR / 'cascade_test_confusion.png')
    plot_confusion(routed_m['confusion_matrix'], CATEGORIES_EN,
                   'Confusion Matrix', 'confusion_matrix.png')

    print('\nClassification report (deployed cascade, predicted routing):')
    print(classification_report(routed['y_true'], routed['y_pred'],
                                target_names=CATEGORIES, digits=4))

    print('\n' + '=' * 80)
    print('Deployed system: coarse 3-way, then S1/S2')
    print('=' * 80)
    print(f'  Coarse 3-way ACC: {coarse_m["accuracy"] * 100:.2f}%')
    print(f'  5-class ACC (predicted routing): {routed_m["accuracy"] * 100:.2f}%')
    print(f'  Macro-F1: {routed_m["macro_f1"]:.4f}')
    print(f'  Kappa: {routed_m["kappa"]:.4f}')
    print(f'  CS: {routed_m["confusion_score"]}  (flat CS={pre_test_m["confusion_score"]})')
    print('  --- diagnostic: oracle in-group (true superclass, not deployed) ---')
    print(f'  S1 ACC (2 vs 3, in-group): {s1_acc:.2f}%')
    print(f'  S2 ACC (0 vs 4, in-group): {s2_acc:.2f}%')
    print(f'  5-class ACC (in-group): {ingroup_m["accuracy"] * 100:.2f}%')
    print('=' * 80)

    results['coarse_test'] = coarse_m
    results['s1_oracle_acc'] = s1_acc
    results['s2_oracle_acc'] = s2_acc
    results['s1_ingroup'] = s1_m
    results['s2_ingroup'] = s2_m
    results['ingroup_test'] = ingroup_m
    results['cascade_test'] = routed_m
    results['predicted_routing'] = routed_m
    results['predicted_routing_acc'] = routed_m['accuracy']
    results['predictions'] = {
        'y_true': routed['y_true'],
        'y_pred': routed['y_pred'],
    }
    with open('atcsa_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\nSaved atcsa_results.json, checkpoints/, figures/')
    print('Training complete.')


if __name__ == '__main__':
    main()
