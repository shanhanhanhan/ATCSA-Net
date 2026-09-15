#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the trained ATCSA cascade on the held-out test split."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from train_classification_model import (
    CATEGORIES,
    CATEGORIES_EN,
    CHECKPOINT_DIR,
    DEVICE,
    FINE_S1_CLASSES,
    FINE_S2_CLASSES,
    FIGURE_DIR,
    SCPDataset,
    cascade_predict,
    compute_metrics,
    load_cascade,
    make_loader,
    oracle_branch_accuracy,
    oracle_ingroup_predict,
    plot_confusion,
)


def load_split_samples(split_path='atcsa_splits.json'):
    with open(split_path, encoding='utf-8') as f:
        payload = json.load(f)
    test_s = [(item['path'], item['label']) for item in payload['test']]
    return test_s, payload


def predict_on_testset(split_path='atcsa_splits.json', save_results=True, save_plots=True):
    print('=' * 80)
    print('ATCSA-Net cascade test evaluation')
    print('=' * 80)
    print(f'Device: {DEVICE}')

    cascade_path = CHECKPOINT_DIR / 'atcsa_coarse.pth'
    if not cascade_path.exists():
        raise FileNotFoundError(
            f'Missing {cascade_path}. Run train_classification_model.py first.'
        )

    test_s, _ = load_split_samples(split_path)
    print(f'\nLoading {len(test_s)} test samples with SCP ...')
    test_ds = SCPDataset(test_s)
    test_loader = make_loader(test_ds, shuffle=False)

    print('Loading cascade models ...')
    model = load_cascade()

    routed = cascade_predict(model, test_loader)
    coarse_m = compute_metrics(routed['coarse_true'], routed['coarse_pred'], 3)
    routed_m = compute_metrics(routed['y_true'], routed['y_pred'], 5)
    out = oracle_ingroup_predict(model, test_loader)
    ingroup_m = compute_metrics(out['y_true'], out['y_pred'], 5)
    s1_acc, _, _ = oracle_branch_accuracy(model, test_loader, FINE_S1_CLASSES, head='s1')
    s2_acc, _, _ = oracle_branch_accuracy(model, test_loader, FINE_S2_CLASSES, head='s2')

    print('\nClassification report (deployed cascade: coarse then S1/S2):')
    print(classification_report(routed['y_true'], routed['y_pred'],
                                target_names=CATEGORIES, digits=4))
    print(f'Coarse 3-way {{0,4}}/{{2,3}}/{{1}} ACC: {coarse_m["accuracy"] * 100:.2f}%')
    print(f'5-class ACC (predicted routing): {routed_m["accuracy"] * 100:.2f}%')
    print(f'Macro-F1: {routed_m["macro_f1"]:.4f}')
    print(f'Kappa: {routed_m["kappa"]:.4f}')
    print(f'CS: {routed_m["confusion_score"]}')
    print('--- diagnostic: oracle in-group (true superclass, not deployed) ---')
    print(f'S1 (2 vs 3) in-group ACC: {s1_acc:.2f}%')
    print(f'S2 (0 vs 4) in-group ACC: {s2_acc:.2f}%')
    print(f'5-class ACC (in-group): {ingroup_m["accuracy"] * 100:.2f}%')

    if save_results:
        results = {
            'predictions': routed['y_pred'],
            'labels': routed['y_true'],
            'categories': CATEGORIES,
            'accuracy': routed_m['accuracy'],
            'metrics': routed_m,
            'coarse': coarse_m,
            'ingroup': ingroup_m,
            's1_oracle_acc': s1_acc,
            's2_oracle_acc': s2_acc,
        }
        with open('test_predictions.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        np.savez('test_predictions.npz',
                 predictions=np.array(routed['y_pred']),
                 labels=np.array(routed['y_true']))
        print('Saved test_predictions.json / test_predictions.npz')

    if save_plots:
        FIGURE_DIR.mkdir(exist_ok=True)
        plot_confusion(routed_m['confusion_matrix'], CATEGORIES_EN,
                       'Test Set Confusion Matrix (predicted routing)',
                       'test_confusion_matrix.png')
        plot_confusion(routed_m['confusion_matrix'], CATEGORIES_EN,
                       'Predicted Routing (5-class)',
                       FIGURE_DIR / 'cascade_test_confusion.png')

        accs = []
        y_true = np.array(routed['y_true'])
        y_pred = np.array(routed['y_pred'])
        for i in range(len(CATEGORIES)):
            mask = y_true == i
            accs.append(float((y_pred[mask] == i).mean()) if mask.any() else 0.0)
        plt.figure(figsize=(10, 6))
        plt.bar(CATEGORIES_EN, accs)
        plt.title('Accuracy by Category')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig('test_category_accuracy.png', dpi=150)
        plt.close()
        print('Saved test_confusion_matrix.png / test_category_accuracy.png')

    print('=' * 80)
    return routed_m


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ATCSA cascade test evaluation')
    parser.add_argument('--split', type=str, default='atcsa_splits.json')
    parser.add_argument('--no-save-results', action='store_true')
    parser.add_argument('--no-save-plots', action='store_true')
    args = parser.parse_args()
    predict_on_testset(
        split_path=args.split,
        save_results=not args.no_save_results,
        save_plots=not args.no_save_plots,
    )
