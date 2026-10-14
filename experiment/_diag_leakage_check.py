#!/usr/env python3
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_controlled as ec
from sklearn.model_selection import StratifiedKFold

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'results', 'bitstream_features_controlled.csv')

def random_cv_auc(Xs, y, seed, permute=False):
    yp = np.random.RandomState(seed).permutation(y) if permute else y
    aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xs, yp):
        if len(np.unique(yp[tr])) < 2 or len(np.unique(yp[te])) < 2:
            continue
        prob = ec.fit_predict(Xs[tr], yp[tr], Xs[te], 'xgb')
        aucs.append(ec.roc_auc_score(yp[te], prob))
    return float(np.mean(aucs))

# load
X, y, feat_cols, codec_arr, src_arr, delta_arr = ec.load_data(CSV)
Xs, _ = ec.select_features(X, feat_cols, ablate=False)
print(f"loaded: X={Xs.shape}, single={int((y==0).sum())}, double={int((y==1).sum())}")

print("\n[1] SOURCE-AWARE XGB (expect ~0.665):")
ec.source_aware_cv(X, y, src_arr, feat_cols, model='xgb', ablate=False)

print("\n[2] RANDOM-CV XGB (mean over 10 seeds):")
vals = [random_cv_auc(Xs, y, s) for s in range(10)]
print(f"     AUC = {np.mean(vals):.4f} +/- {1.96*np.std(vals)/np.sqrt(len(vals)):.4f}")

print("\n[3] RANDOM-CV XGB with PERMUTED labels (expect ~0.50):")
vals_p = [random_cv_auc(Xs, y, s, permute=True) for s in range(10)]
print(f"     AUC = {np.mean(vals_p):.4f} +/- {1.96*np.std(vals_p)/np.sqrt(len(vals_p)):.4f}")

"""Diagnostic: compare source-aware CV against random CV."""

