#!/usr/env python3
# -*- coding: utf-8 -*-
"""
Random (stratified, leakage-prone) CV — the "legacy" protocol demonstration.
=====================================================================
Reuses load_data / select_features / fit_predict / metrics from
eval_controlled.py so the feature set and the threshold protocol are
IDENTICAL to the source-aware run. The ONLY difference is that here we
use StratifiedKFold (which IGNORES the source video), so the same raw
source's single & double clips can land on opposite sides of the
train/test split — i.e. the content-leakage artifact.

Usage:
  py eval_random_cv.py --features results/bitstream_features_controlled.csv --model xgb
  py eval_random_cv.py --features results/bitstream_features_controlled.csv --model lr
"""
import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

# import the controlled-eval machinery (same features, same metrics)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_controlled as ec   # noqa: E402

SEEDS = 10
N_FOLDS = 5


def random_cv(features_csv, model='xgb', ablate=False):
    X, y, feat_cols, codec_arr, src_arr, delta_arr = ec.load_data(features_csv)
    Xs, _ = ec.select_features(X, feat_cols, ablate)
    aucs, eers, td01, td05 = [], [], [], []
    for seed in range(SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xs, y):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            prob = ec.fit_predict(Xs[tr], y[tr], Xs[te], model)
            prob_tr = ec.fit_predict(Xs[tr], y[tr], Xs[tr], model)
            aucs.append(ec.roc_auc_score(y[te], prob))
            eers.append(ec.eer(y[te], prob))
            td01.append(ec.tdr_at_fpr(y[tr], prob_tr, y[te], prob, 0.01))
            td05.append(ec.tdr_at_fpr(y[tr], prob_tr, y[te], prob, 0.05))
    tag = "FULL" if not ablate else "NO-RATE"
    print(f"\n  [{tag}] RANDOM(stratified, leaky) {N_FOLDS}-fold x {SEEDS} seeds ({model})")
    print(f"      AUC : {ec.ci_str(aucs)}")
    print(f"      EER : {ec.ci_str(eers)}")
    print(f"      TDR@FPR=0.01 : {ec.ci_str(td01)}")
    print(f"      TDR@FPR=0.05 : {ec.ci_str(td05)}")
    return {'auc': ec.ci_str(aucs), 'eer': ec.ci_str(eers),
            'tdr01': ec.ci_str(td01), 'tdr05': ec.ci_str(td05)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=str, required=True)
    ap.add_argument('--model', type=str, default='xgb', choices=['lr', 'xgb'])
    args = ap.parse_args()
    random_cv(args.features, model=args.model, ablate=False)


if __name__ == '__main__':
    main()
