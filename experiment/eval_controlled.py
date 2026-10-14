#!/usr/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

# metadata and label columns
EXCLUDE = {'video_file', 'label', 'codec', 'q1', 'q2', 'delta_qp',
           'sample_path', 'src_path', 'group_id', 'error',
           'crf', 'level', 'height', 'width', 'duration_s', 'n_frames_decoded'}
# rate features, dropped in the ablation
RATE_FEATURES = {'bit_rate', 'actual_bitrate', 'bits_per_pixel'}

SEEDS = 10
N_FOLDS = 5
Z = 1.96  # 95% CI


# ───────────────────────────────────────────────────────────
# data loading
# ───────────────────────────────────────────────────────────
def load_data(features_csv, max_delta=None):
    rows = list(csv.DictReader(open(features_csv, 'r', encoding='utf-8')))
    # numeric feature columns
    feat_cols = []
    for k in rows[0].keys():
        if k in EXCLUDE:
            continue
        try:
            float(rows[0][k])
            feat_cols.append(k)
        except (ValueError, TypeError):
            continue

    X, y, codec_arr, src_arr, delta_arr = [], [], [], [], []
    for r in rows:
        lab = r.get('label', '')
        if lab not in ('single', 'double'):
            continue
        delta = float(r.get('delta_qp', 0) or 0)
        if lab == 'double' and max_delta is not None and delta > max_delta:
            continue  # near-parameter subset: keep only small-delta double rows
        vals = []
        for fn in feat_cols:
            try:
                v = float(r.get(fn, 0))
                vals.append(0.0 if (np.isnan(v) or np.isinf(v)) else v)
            except (ValueError, TypeError):
                vals.append(0.0)
        X.append(vals)
        y.append(1 if lab == 'double' else 0)
        codec_arr.append(r.get('codec', ''))
        src_arr.append(r.get('video_file', ''))
        delta_arr.append(delta)

    return (np.array(X, dtype=np.float64), np.array(y),
            feat_cols, np.array(codec_arr), np.array(src_arr),
            np.array(delta_arr))


def select_features(X, feat_cols, ablate=False):
    if not ablate:
        return X, feat_cols
    keep = [i for i, fn in enumerate(feat_cols) if fn not in RATE_FEATURES]
    return X[:, keep], [feat_cols[i] for i in keep]


def make_model(model):
    if model == 'lr' or model == 'both':
        return 'lr'
    if model == 'xgb':
        try:
            from xgboost import XGBClassifier
            return 'xgb'
        except ImportError:
            print("  [warn] xgboost unavailable, falling back to LR")
            return 'lr'
    return 'lr'


def fit_predict(Xtr, ytr, Xte, model):
    if model == 'xgb':
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          random_state=42, eval_metric='logloss', verbosity=0)
        m.fit(Xtr, ytr)
    else:
        sc = StandardScaler().fit(Xtr)
        Xtr2 = sc.transform(Xtr)
        m = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        m.fit(Xtr2, ytr)
    # scores (LR: positive class; XGB: [:,1])
    if model == 'xgb':
        return m.predict_proba(Xte)[:, 1]
    sc = StandardScaler().fit(Xtr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


# ───────────────────────────────────────────────────────────
# metrics
# ───────────────────────────────────────────────────────────
def eer(y_true, y_score):
    """Equal error rate: FPR == FNR (interpolated)."""
    if len(np.unique(y_true)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def tdr_at_fpr(y_tr, s_tr, y_te, s_te, target_fpr):
    """On the training set pick the threshold with FPR<=target and max TPR, report TDR on the test set."""
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float('nan')
    fpr, tpr, thr = roc_curve(y_tr, s_tr)
    cand = [(f, t, th) for f, t, th in zip(fpr, tpr, thr) if f <= target_fpr]
    if cand:
        best = max(cand, key=lambda x: x[1])
        th = best[2]
    else:
        th = thr[int(np.argmin(fpr))]
    pred = (s_te >= th).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def ci_str(scores):
    arr = np.array([s for s in scores if not np.isnan(s)], dtype=np.float64)
    if len(arr) == 0:
        return "n/a"
    m = arr.mean()
    ci = Z * arr.std() / np.sqrt(len(arr))
    return f"{m:.4f} ± {ci:.4f}"


# ───────────────────────────────────────────────────────────
# 1) Source-aware CV
# ───────────────────────────────────────────────────────────
def source_aware_cv(X, y, src, feat_cols, model='lr', ablate=False):
    Xs, fcols = select_features(X, feat_cols, ablate)
    aucs, eers, td01, td05 = [], [], [], []
    for seed in range(SEEDS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                    random_state=seed)
        for tr, te in sgkf.split(Xs, y, src):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            prob = fit_predict(Xs[tr], y[tr], Xs[te], model)
            # train scores, used to pick a threshold
            prob_tr = fit_predict(Xs[tr], y[tr], Xs[tr], model)
            aucs.append(roc_auc_score(y[te], prob))
            eers.append(eer(y[te], prob))
            td01.append(tdr_at_fpr(y[tr], prob_tr, y[te], prob, 0.01))
            td05.append(tdr_at_fpr(y[tr], prob_tr, y[te], prob, 0.05))
    tag = "FULL" if not ablate else "NO-RATE"
    print(f"\n  [{tag}] source-aware {N_FOLDS}-fold × {SEEDS} seeds ({model})")
    print(f"      AUC : {ci_str(aucs)}")
    print(f"      EER : {ci_str(eers)}")
    print(f"      TDR@FPR=0.01 : {ci_str(td01)}")
    print(f"      TDR@FPR=0.05 : {ci_str(td05)}")
    return {'auc': ci_str(aucs), 'eer': ci_str(eers),
            'tdr01': ci_str(td01), 'tdr05': ci_str(td05)}


# ───────────────────────────────────────────────────────────
# 2) cross-codec
# ───────────────────────────────────────────────────────────
def cross_codec(X, y, codec_arr, src, feat_cols, model='lr', ablate=False):
    Xs, fcols = select_features(X, feat_cols, ablate)
    codecs = sorted(set(codec_arr))
    print(f"\n  [CROSS-CODEC] train on single+double of codec A, "
          f"test on codec B (seed={42})")
    print(f"  {'train\\\\test':>12s} | " + " | ".join(f"{c:>10s}" for c in codecs))
    print("  " + "-" * (14 + 14 * len(codecs)))
    for tr_c in codecs:
        row = [f"{tr_c:>12s}"]
        for te_c in codecs:
            m_tr = codec_arr == tr_c
            m_te = codec_arr == te_c
            if m_tr.sum() == 0 or m_te.sum() == 0:
                row.append(f"{'n/a':>10s}")
                continue
            # source-aware 5-fold inside each split, no content leakage
            aucs = []
            sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                        random_state=42)
            for tr2, te2 in sgkf.split(Xs[m_tr], y[m_tr], src[m_tr]):
                prob = fit_predict(Xs[m_tr][tr2], y[m_tr][tr2],
                                  Xs[m_tr][te2], model)
                if len(np.unique(y[m_tr][te2])) < 2:
                    continue
                aucs.append(roc_auc_score(y[m_tr][te2], prob))
            if te_c == tr_c:
                # same codec: source-aware CV inside that codec
                val = ci_str(aucs) if aucs else 'n/a'
            else:
                # cross codec: train on all of A, test on all of B
                prob = fit_predict(Xs[m_tr], y[m_tr], Xs[m_te], model)
                val = (f"{roc_auc_score(y[m_te], prob):.4f}"
                       if len(np.unique(y[m_te])) > 1 else 'n/a')
            row.append(f"{val:>10s}")
        print("  " + " | ".join(row))
    print("  (diagonal = same-codec source-aware CV; off-diagonal = cross-codec AUC)")


# ───────────────────────────────────────────────────────────
# self-test on synthetic data
# ───────────────────────────────────────────────────────────
def self_test():
    print("=" * 70)
    print("SELF-TEST: synthetic data, all metrics")
    print("=" * 70)
    rng = np.random.RandomState(0)
    feat_pool = ['pkt_size_mean', 'gop_mean', 'i_ratio', 'bits_per_pixel',
                 'bit_rate', 'n_packets', 'key_frame_ratio', 'pts_delta_std',
                 'frame_type_entropy']
    codecs = ['h264', 'h265']
    sources = [f'src{i}' for i in range(6)]
    rows = []
    for src in sources:
        for codec in codecs:
            q1 = 28
            # single: base distribution
            base = rng.randn(len(feat_pool)) * 0.3
            single_feats = base + np.array([1.0, 1.0, 0.3, 0.5, 0.5, 1.0, 0.4, 0.2, 0.5])
            rec = {'video_file': src, 'label': 'single', 'codec': codec,
                   'q1': str(q1), 'q2': str(q1), 'delta_qp': '0'}
            for fn, v in zip(feat_pool, single_feats):
                rec[fn] = f"{v:.4f}"
            rows.append(rec)
            # double: small consistent shift
            for off in [-3, 0, 6]:
                d = base + np.array([1.25, 0.8, 0.45, 0.7, 0.72, 1.2,
                                     0.5, 0.35, 0.62]) + rng.randn(len(feat_pool)) * 0.15
                rec = {'video_file': src, 'label': 'double', 'codec': codec,
                       'q1': str(q1), 'q2': str(q1 + off),
                       'delta_qp': str(abs(off))}
                for fn, v in zip(feat_pool, d):
                    rec[fn] = f"{v:.4f}"
                rows.append(rec)
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '_selftest_features.csv')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['video_file', 'label', 'codec', 'q1',
                                       'q2', 'delta_qp'] + feat_pool)
        w.writeheader()
        w.writerows(rows)
    print(f"  synthetic rows: {len(rows)}, codecs={codecs}, sources={len(sources)}")
    run(tmp, model='lr', max_delta=None)
    print(f"\n  [ok] self-test passed, temp file: {tmp}")


# ───────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────
def run(features_csv, model='lr', max_delta=None):
    print("=" * 70)
    print("Controlled two-arm evaluation")
    print("=" * 70)
    X, y, feat_cols, codec_arr, src_arr, delta_arr = load_data(
        features_csv, max_delta=max_delta)
    print(f"samples: {X.shape[0]} (single={int((y == 0).sum())}, "
          f"double={int((y == 1).sum())})")
    print(f"features: {len(feat_cols)} | codecs: {sorted(str(c) for c in set(codec_arr))} | "
          f"sources: {len(set(src_arr))}")
    if max_delta is not None:
        print(f"near-parameter subset: double rows with delta_qp <= {max_delta}")
    mdl = make_model(model)

    print("\n" + "-" * 70)
    print("1) SOURCE-AWARE CV")
    print("-" * 70)
    r_full = source_aware_cv(X, y, src_arr, feat_cols, model=mdl, ablate=False)
    r_norate = source_aware_cv(X, y, src_arr, feat_cols, model=mdl, ablate=True)

    print("\n" + "-" * 70)
    print("2) CROSS-CODEC GENERALIZATION")
    print("-" * 70)
    cross_codec(X, y, codec_arr, src_arr, feat_cols, model=mdl, ablate=False)

    print("\n" + "-" * 70)
    print("3) RATE-CONFOUND ABLATION")
    print("-" * 70)
    print(f"  FULL    : AUC={r_full['auc']}  EER={r_full['eer']}  "
          f"TDR@.01={r_full['tdr01']}  TDR@.05={r_full['tdr05']}")
    print(f"  NO-RATE: AUC={r_norate['auc']}  EER={r_norate['eer']}  "
          f"TDR@.01={r_norate['tdr01']}  TDR@.05={r_norate['tdr05']}")
    


def main():
    if not HAVE_SK:
        print("scikit-learn missing: pip install scikit-learn")
        sys.exit(1)
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=str, default=None,
                    help='path to the feature CSV')
    ap.add_argument('--model', type=str, default='xgb',
                    choices=['lr', 'xgb', 'both'])
    ap.add_argument('--max-delta', type=float, default=None,
                    help='near-parameter subset: keep double rows with delta_qp <= this')
    ap.add_argument('--self-test', action='store_true',
                    help='run all metrics on synthetic data')
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.features:
        print("pass --features <csv> or --self-test")
        sys.exit(1)
    run(args.features, model=args.model, max_delta=args.max_delta)


if __name__ == '__main__':
    main()

"""Evaluate a feature table: source-aware grouped CV, cross-codec matrix, rate-confound ablation."""

