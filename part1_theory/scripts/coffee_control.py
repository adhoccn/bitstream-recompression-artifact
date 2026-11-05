# -*- coding: utf-8 -*-
"""
CoFFEE-only control arm for Part II, Section XI.

Purpose
-------
Part II reports a pixel-domain negative result (per-stage AUC 0.4688-0.5111).
What it lacks is a coded-layer contrast: a feature set read from the quantized
syntax elements, before dequantization and in-loop filtering. CoFFEE
(Bertazzini et al., EURASIP J. Inf. Secur. 2024) supplies exactly that, and it
is open source. The corpus here is a separate H.264 subset, not the 198-sample
pixel-domain corpus; the paper states that limit explicitly.

This script deliberately contains NO in-house bitstream arm. The sample
inventory, the labels and the group ids are all derived from the CoFFEE
feature files alone, so the numbers produced here carry a clean provenance
and can be reported in Part II without touching any other manuscript's
material. Set --check-csv to additionally verify (read-only, inventory only,
no feature values) that the 100 stems coincide with another pipeline's list.

Protocol
--------
  StratifiedGroupKFold(5, shuffle, random_state=seed), seed in 0..9
  XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, eval_metric='logloss')
  per-fold AUC / EER / TDR@FPR<=0.05 (threshold chosen on the TRAIN fold)
  CI = 1.96 * sd / sqrt(n) over the 50 fold estimates

Arms (column blocks of the 72125-D CoFFEE Sec. 5.1 vector)
---------------------------------------------------------
  B  coffee_full   0 .. 72125
  C  coffee_dct    0 .. 72000        AC-DCT coefficient histograms
  D  coffee_mb     72000 .. 72021    macroblock-type statistics
  E  coffee_qp     72021 .. 72125    QP histograms

Subsets
-------
  full     all 100 clips (20 single + 80 double), 20 sources
  delta0   single + the q2 = q1 recompression   (20 + 20)
  delta3   single + the q2 = q1-3 recompression (20 + 20)

Data
----
The 72125-D CoFFEE vectors are not redistributed here: they are large binary
dumps derived from the corpus. Regenerate them with the CoFFEE roaster
(github.com/IAPP-Group/CoFFEE) on the same 100 H.264 clips, one ``.npy`` per
clip named ``{single|double}__{source}_{suffix}.npy``, then point
``--features-dir`` at that directory. The archived
``data/coffee_control_arms.json`` holds the aggregate results this script
reproduces; per-fold vectors are omitted there to keep the file small and are
regenerated deterministically by re-running the script.

Usage
-----
  python run_coffee_arms.py                     # all arms, full subset
  python run_coffee_arms.py --subset delta3     # near-parameter
  python run_coffee_arms.py --arm E_coffee_qp   # single arm
  python run_coffee_arms.py --check-csv PATH    # + inventory cross-check
  python run_coffee_arms.py --seeds 2           # smoke test
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
CTRL_ROOT = os.path.dirname(HERE)                       # .../companion_theory
WORKSPACE = os.path.dirname(CTRL_ROOT)
FEAT_DIR = os.environ.get('COFFEE_FEATURES',
                          os.path.join(WORKSPACE, 'coffee_features'))

D_NAMES = ['B_coffee_full', 'C_coffee_dct', 'D_coffee_mb', 'E_coffee_qp']
BLOCKS = {
    'B_coffee_full': slice(0, 72125),
    'C_coffee_dct': slice(0, 72000),
    'D_coffee_mb': slice(72000, 72021),
    'E_coffee_qp': slice(72021, 72125),
}
Z = 1.96


# ----------------------------------------------------------------- inventory
def load_inventory():
    """Derive (label, stem) -> feature vector, strictly from CoFFEE files."""
    files = sorted(glob.glob(os.path.join(FEAT_DIR, '*.npy')))
    if not files:
        sys.exit('no .npy under %s' % FEAT_DIR)
    data, dims = {}, collections.Counter()
    for f in files:
        name = os.path.basename(f)[:-4]
        if '__' not in name:
            sys.exit('unexpected feature filename (no "__"): %s' % name)
        label, stem = name.split('__', 1)
        if label not in ('single', 'double'):
            sys.exit('unexpected label prefix: %s' % name)
        arr = np.load(f).astype(np.float64)
        dims[arr.shape] += 1
        data[(label, stem)] = arr
    if len(dims) != 1:
        sys.exit('ragged feature files: %s' % dict(dims))
    width = next(iter(dims))
    if width[0] != 72125:
        sys.exit('expected 72125 columns, got %s' % (width,))
    return data


def check_against_csv(keys, csv_path):
    """Inventory-only cross-check. Reads video_file/label columns; no features."""
    import csv as _csv
    if not os.path.exists(csv_path):
        print('  [check] csv not found, skipped: %s' % csv_path)
        return
    stems = set()
    with open(csv_path, encoding='utf-8') as fh:
        for row in _csv.DictReader(fh):
            if row.get('codec') not in (None, '', 'h264'):
                continue
            sp = row.get('sample_path') or ''
            stems.add(os.path.basename(sp).replace('.mp4', ''))
    mine = set(k[1] for k in keys)
    inter, only_mine, only_theirs = mine & stems, mine - stems, stems - mine
    print('  [check] stems here=%d csv=%d matched=%d only_here=%d only_csv=%d'
          % (len(mine), len(stems), len(inter), len(only_mine), len(only_theirs)))
    if only_mine:
        print('  [check] only here:', sorted(only_mine)[:5])
    if only_theirs:
        print('  [check] only csv :', sorted(only_theirs)[:5])


def subset_keys(keys, subset):
    if subset == 'full':
        return keys
    suffix = {'delta3': '_q225', 'delta0': '_q228'}[subset]
    out = [k for k in keys if k[0] == 'single' or k[1].endswith(suffix)]
    return out


# ------------------------------------------------------------------- metrics
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def tdr05(ytr, str_, yte, ste):
    fpr, tpr, thr = roc_curve(ytr, str_)
    cand = [(f, t, th) for f, t, th in zip(fpr, tpr, thr) if f <= 0.05]
    th = max(cand, key=lambda x: x[1])[2] if cand else thr[int(np.argmin(fpr))]
    pred = ste >= th
    tp = int(((pred == 1) & (yte == 1)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    return tp / (tp + fn) if tp + fn else 0.0


def ci(v):
    return (float(np.mean(v)), float(Z * np.std(v) / np.sqrt(len(v))))


# ---------------------------------------------------------------------- arms
def run_arm(X, y, src, arm, seeds, cache, cache_path, tag):
    aucs, eers, tdrs, folds = [], [], [], {}
    for seed in range(seeds):
        ck = '%s__%s__%d' % (tag, arm, seed)
        if ck in cache:
            e = cache[ck]
            aucs += e['aucs']; eers += e['eers']; tdrs += e['tdrs']
            folds.update(e['fold_aucs'])
            continue
        e = {'aucs': [], 'eers': [], 'tdrs': [], 'fold_aucs': {}}
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fi, (tr, te) in enumerate(sgkf.split(X, y, src)):
            m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                              random_state=42, eval_metric='logloss', verbosity=0,
                              n_jobs=8)
            m.fit(X[tr], y[tr])
            prob = m.predict_proba(X[te])[:, 1]
            prob_tr = m.predict_proba(X[tr])[:, 1]
            a = float(roc_auc_score(y[te], prob))
            e['aucs'].append(a)
            e['eers'].append(eer(y[te], prob))
            e['tdrs'].append(tdr05(y[tr], prob_tr, y[te], prob))
            e['fold_aucs']['%d_%d' % (seed, fi)] = a
        cache[ck] = e
        json.dump(cache, open(cache_path, 'w'), indent=1)
        print('    seed %d  mean AUC %.4f' % (seed, np.mean(e['aucs'])), flush=True)
        aucs += e['aucs']; eers += e['eers']; tdrs += e['tdrs']
        folds.update(e['fold_aucs'])
    out = {'auc': ci(aucs), 'eer': ci(eers), 'tdr05': ci(tdrs),
           'n_folds': len(aucs), 'fold_aucs': folds}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='full', choices=['full', 'delta0', 'delta3'])
    ap.add_argument('--arm', default=None, choices=D_NAMES)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--check-csv', default=None)
    ap.add_argument('--features-dir', default=None,
                    help='directory of CoFFEE 72125-D .npy files '
                         '(default: $COFFEE_FEATURES or ../coffee_features)')
    args = ap.parse_args()

    global FEAT_DIR
    if args.features_dir:
        FEAT_DIR = args.features_dir
    tag = args.subset
    out_path = os.path.join(HERE, 'coffee_arms_%s.json' % tag)
    cache_path = os.path.join(HERE, '_cache_%s.json' % tag)

    cf = load_inventory()
    keys = sorted(cf)
    print('CoFFEE inventory: %d clips, %d sources, dims=%s'
          % (len(keys), len(set(re.sub(r'\D.*$', '', k[1]) for k in keys)),
             next(iter(cf.values())).shape))
    if args.check_csv:
        check_against_csv(keys, args.check_csv)

    keys = subset_keys(keys, tag)
    y = np.array([1 if k[0] == 'double' else 0 for k in keys])
    src = np.array([re.sub(r'\D.*$', '', k[1]) for k in keys])
    print('subset=%s  clips=%d  label balance=%s  sources=%d'
          % (tag, len(keys), dict(zip(*np.unique(y, return_counts=True))),
             len(set(src))))
    if len(set(y)) < 2:
        sys.exit('subset has a single class')

    X_all = np.stack([cf[k] for k in keys])
    print('X shape:', X_all.shape, flush=True)

    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    results.setdefault('_meta', {
        'subset': tag, 'clips': len(keys), 'sources': int(len(set(src))),
        'label_balance': {str(int(k)): int(v)
                          for k, v in zip(*np.unique(y, return_counts=True))},
        'seeds': args.seeds,
        'protocol': ('StratifiedGroupKFold(5,shuffle)x seeds 0..N-1, groups=source; '
                     'XGB(100,d3,lr0.1,rs42); TDR@FPR<=0.05 train-fold threshold; '
                     'CI=1.96*sd/sqrt(n_folds)'),
        'feature_source': 'CoFFEE (Bertazzini et al. 2024), 72125-D Sec.5.1 vector',
        'in_house_arm': 'absent by design',
    })

    for arm in (D_NAMES if args.arm is None else [args.arm]):
        X = X_all[:, BLOCKS[arm]]
        print('--- %s (%d dims) ...' % (arm, X.shape[1]), flush=True)
        r = run_arm(X, y, src, arm, args.seeds, cache, cache_path, tag)
        results[arm] = r
        json.dump(results, open(out_path, 'w'), indent=1)
        print('    AUC %.4f +- %.4f | EER %.4f +- %.4f | TDR@.05 %.4f +- %.4f'
              % (r['auc'][0], r['auc'][1], r['eer'][0], r['eer'][1],
                 r['tdr05'][0], r['tdr05'][1]), flush=True)
        print('    saved', out_path, flush=True)


if __name__ == '__main__':
    main()
