#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLIND_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(BLIND_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

DEFAULT_FEATURES = os.path.join(BLIND_DIR, "results",
                                  "bitstream_features_controlled.csv")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, SCRIPT_DIR)
from eval_controlled import (load_data, select_features, fit_predict,
                          SEEDS, N_FOLDS)
from sklearn.model_selection import StratifiedGroupKFold


def collect_scores(features_csv):
    """Out-of-fold XGB test scores under source-aware CV (full 300-clip)."""
    X, y, feat_cols, codec_arr, src_arr, delta_arr = load_data(features_csv)
    Xs, _ = select_features(X, feat_cols, ablate=False)
    scores, labels = [], []
    for seed in range(SEEDS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                     random_state=seed)
        for tr, te in sgkf.split(Xs, y, src_arr):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            prob = fit_predict(Xs[tr], y[tr], Xs[te], "xgb")
            scores.extend(prob.tolist())
            labels.extend(y[te].tolist())
    return np.array(scores), np.array(labels, dtype=int)


def fig_scores(scores, labels):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bins = np.linspace(0, 1, 31)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, density=True,
            label="single (y=0)", color="#4C72B0")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, density=True,
            label="double (y=1)", color="#DD8452")
    ax.set_xlabel("XGBoost double-compression score (out-of-fold)")
    ax.set_ylabel("density")
    ax.set_title("Source-aware CV: score distributions (single vs double)")
    ax.legend(loc="upper center")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_scores.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_importance(features_csv):
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y, feat_cols, _, _, _ = load_data(features_csv)
    Xs, fcols = select_features(X, feat_cols, ablate=False)

    # XGB gain (map internal fNN indices back to named features)
    m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                      random_state=42, verbosity=0).fit(Xs, y)
    gain = m.get_booster().get_score(importance_type="gain")
    tot_g = sum(gain.values()) or 1.0
    def _name(k):
        if isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
            return feat_cols[int(k[1:])]
        return k

    g_items = sorted(
        ((_name(k), v / tot_g) for k, v in gain.items()),
        key=lambda kv: -kv[1])[:10]

    # LR standardized |coef|
    sc = StandardScaler().fit(Xs)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(sc.transform(Xs), y)
    coef = np.abs(lr.coef_[0])
    order = np.argsort(-coef)[:10]
    lr_items = [(fcols[i], coef[i]) for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    # left: XGB gain
    names = [k for k, _ in g_items][::-1]
    vals = [v for _, v in g_items][::-1]
    axes[0].barh(range(len(names)), vals, color="#4C72B0")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].set_xlabel("normalized gain")
    axes[0].set_title("XGBoost (gain, top-10)", fontsize=9)
    # right: LR coef
    names2 = [k for k, _ in lr_items][::-1]
    vals2 = [v for _, v in lr_items][::-1]
    axes[1].barh(range(len(names2)), vals2, color="#55A868")
    axes[1].set_yticks(range(len(names2)))
    axes[1].set_yticklabels(names2, fontsize=7)
    axes[1].set_xlabel("|standardized weight|")
    axes[1].set_title("Logistic Regression (top-10)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_importance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, g_items, lr_items


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEATURES
    print(f"Loading {csv_path}")
    scores, labels = collect_scores(csv_path)
    auc_proxy = ((scores[labels == 1].mean() > scores[labels == 0].mean()))
    print(f"  collected {len(scores)} out-of-fold scores; "
          f"double-mean={scores[labels == 1].mean():.3f} "
          f"single-mean={scores[labels == 0].mean():.3f}")
    f1 = fig_scores(scores, labels)
    print(f"  wrote {f1}")
    f2, g_items, lr_items = fig_importance(csv_path)
    print(f"  wrote {f2}")
    print("  XGB gain top-3:", [(f, round(v, 3)) for f, v in g_items[:3]])
    print("  LR |coef| top-3:", [(f, round(v, 3)) for f, v in lr_items[:3]])


if __name__ == "__main__":
    main()

"""Generate the two summary figures from a feature table."""

