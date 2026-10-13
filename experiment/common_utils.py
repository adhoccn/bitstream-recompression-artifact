#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blind-GCRR 公共工具模块

提供评估指标计算、CSV读写、置信区间等共享功能。
"""

import csv
import os
import numpy as np
from typing import List, Dict, Tuple


def read_sample_index(csv_path: str) -> List[Dict]:
    """读取样本索引CSV"""
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def compute_metrics(y_true, y_pred, y_scores, threshold):
    """计算分类指标"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'threshold': threshold,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def calibrate_threshold_youden(labels, scores):
    """
    用 Youden's J statistic 标定最优阈值
    J = TPR - FPR，最大化J得到最优阈值
    """
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # 使用唯一分数作为候选阈值
    unique_scores = np.unique(scores_arr)
    if len(unique_scores) > 100:
        candidates = np.percentile(scores_arr, np.linspace(1, 99, 99))
    else:
        candidates = unique_scores

    best_j = -1
    best_threshold = 0.5
    best_metrics = None

    for t in candidates:
        pred = (scores_arr > t).astype(int)
        tp = np.sum((pred == 1) & (labels_arr == 1))
        fp = np.sum((pred == 1) & (labels_arr == 0))
        tn = np.sum((pred == 0) & (labels_arr == 0))
        fn = np.sum((pred == 0) & (labels_arr == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        j = tpr - fpr

        if j > best_j:
            best_j = j
            best_threshold = t
            best_metrics = {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
                           'tpr': float(tpr), 'fpr': float(fpr), 'j': float(j)}

    return best_threshold, best_metrics


def bootstrap_ci(y_true, y_scores, threshold, metric='f1', n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap 95%置信区间"""
    n = len(y_true)
    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)

    bootstrap_metrics = []
    rng = np.random.RandomState(seed)

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        bt_true = y_true_arr[indices]
        bt_scores = y_scores_arr[indices]
        bt_pred = (bt_scores > threshold).astype(int)

        tp = np.sum((bt_true == 1) & (bt_pred == 1))
        fp = np.sum((bt_true == 0) & (bt_pred == 1))
        tn = np.sum((bt_true == 0) & (bt_pred == 0))
        fn = np.sum((bt_true == 1) & (bt_pred == 0))

        if metric == 'accuracy':
            val = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        elif metric == 'precision':
            val = tp / (tp + fp) if (tp + fp) > 0 else 0
        elif metric == 'recall':
            val = tp / (tp + fn) if (tp + fn) > 0 else 0
        elif metric == 'f1':
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        else:
            val = 0

        bootstrap_metrics.append(val)

    bootstrap_metrics = np.array(bootstrap_metrics)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(bootstrap_metrics, alpha * 100))
    upper = float(np.percentile(bootstrap_metrics, (1 - alpha) * 100))

    return lower, upper


def write_results_csv(csv_path, fieldnames, rows):
    """写入结果CSV"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Results saved: {csv_path}")


def format_summary(stage_name, metrics, ci_results, threshold, extra_info=None):
    """格式化摘要文本"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Blind-GCRR {stage_name} Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Threshold (Youden's J): {threshold:.4f}")
    lines.append(f"  TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    lines.append("")

    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        ci_lo, ci_hi = ci_results.get(metric_name, (0, 0))
        lines.append(f"  {metric_name:>10s}: {metrics[metric_name]:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    lines.append("")

    if extra_info:
        for key, value in extra_info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    return '\n'.join(lines)
