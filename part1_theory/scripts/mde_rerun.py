#!/usr/bin/env python3
"""Minimum detectable effect (MDE) for the pixel-domain rerun audit.

Design: 50 source clips x 4 rows (1 single-compressed H0 + 3 double-compressed
H1: h264, h265, vp9), n = 200.  Per-sample scores archived in
data/pixel_domain_rerun.csv.

Method
------
Two variances matter and they are different:

* Independence floor (Hanley-McNeil null variance at AUC = 0.5, Q1 = Q2 = 1/3):
  the optimistic bound that ignores clustering entirely.

* Cluster-bootstrap SE: resample the 50 clips with replacement (duplicated
  clips carry their whole within-clip configuration, so the corpus's
  clustering and cross-arm content correlation are preserved) and, to realise
  H0, choose uniformly which of the 4 rows of each resampled clip is the
  pseudo-H0.  The std of the resulting AUC distribution estimates the
  *sampling* variance of the AUC under H0 for this design -- the quantity
  both the test and the MDE need.  A conditional permutation null would give
  the wrong (narrower) variance here.

MDE: bisection on the shift delta added to the 3 H1 rows of every clip until
the clip-bootstrap rejection rate at the two-sided 5% level reaches 0.80;
report delta and the corresponding population AUC, AUC(delta) = P(S0 < S1+delta)
evaluated on the full observed sample.  Stage 2 excluded: rerun scores are
identically zero (degenerate).
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))


def _archived(name):
    """Locate an archived result file, whichever layout we are in.

    The sibling scripts write to ``<repo>/data/``, which .gitignore drops; the
    files that back a published number live in ``part1_theory/data/``. Readers
    therefore check the archived tree first and the working tree second.
    """
    for rel in (("data", name), (os.pardir, "data", name)):
        path = os.path.join(HERE, *rel)
        if os.path.exists(path):
            return os.path.normpath(path)
    raise SystemExit(
        "mde_rerun: cannot find %s under %s or its parent" % (name, HERE))


CSV = _archived("pixel_domain_rerun.csv")

Z975 = 1.959963985
ALPHA_POWER = 0.80
B_SE = 4000
B_POWER = 2000
SEED = 20261102
STAGES = ["S1", "S3", "S4", "S_fused"]  # S2 excluded: identically zero
LABELS = np.tile([0, 1, 1, 1], 50)      # row 0 of each clip is H0


def stage_matrix(rows: list[dict], clips: list[str], stage: str) -> np.ndarray:
    idx = {c: i for i, c in enumerate(clips)}
    X = np.full((len(clips), 4), np.nan)
    for r in rows:
        j = 0 if r["label"] == "0" else 1 + ["h264", "h265", "vp9"].index(r["codec"])
        X[idx[r["source"]], j] = float(r[stage])
    assert not np.isnan(X).any()
    return X


def auc(vals: np.ndarray, lab: np.ndarray) -> float:
    r = rankdata(vals)
    n1 = int(lab.sum())
    n0 = lab.size - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def boot_null_auc(X: np.ndarray, rng: np.random.Generator, b: int) -> np.ndarray:
    """Resample clips with replacement; per resampled clip pick 1 of 4 rows as H0."""
    nc = X.shape[0]
    out = np.empty(b)
    for k in range(b):
        pick = rng.integers(0, nc, size=nc)
        Y = X[pick]
        h0 = rng.integers(0, 4, size=nc)
        lab = np.ones(4 * nc, dtype=int)
        lab[np.arange(nc) * 4 + h0] = 0
        out[k] = auc(Y.reshape(-1), lab)
    return out


def population_auc(X: np.ndarray, delta: float) -> float:
    """AUC of the location-shift alternative on the full observed sample."""
    return auc(np.concatenate([X[:, 0], X[:, 1:].reshape(-1) + delta]),
               np.array([0] * 50 + [1] * 150))


def power_at(X: np.ndarray, delta: float, crit: float,
             rng: np.random.Generator, b: int = B_POWER) -> float:
    """Clip-bootstrap power at shift delta; reject when AUC > crit."""
    nc = X.shape[0]
    hits = 0
    for _ in range(b):
        Y = X[rng.integers(0, nc, size=nc)].copy()
        Y[:, 1:] += delta
        hits += auc(Y.reshape(-1), LABELS) > crit
    return hits / b


def icc1_h1_arm(X: np.ndarray) -> float:
    Y = X[:, 1:]
    k = Y.shape[1]
    grand = Y.mean()
    msb = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (Y.shape[0] - 1)
    msw = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum() / (Y.shape[0] * (k - 1))
    if msb + msw == 0:
        return float("nan")
    return (msb - msw) / (msb + (k - 1) * msw)


def mde_by_bisection(X: np.ndarray, crit: float, rng: np.random.Generator,
                     lo: float = 0.0, hi: float = 1.0) -> float:
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        if power_at(X, mid, crit, rng) < ALPHA_POWER:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    rows = list(csv.DictReader(io.open(CSV, encoding="utf-8")))
    clips = sorted({r["source"] for r in rows})
    n0, n1 = 50, 150
    rng = np.random.default_rng(SEED)

    var_ind = (0.25 + (n1 - 1) * (1 / 12) + (n0 - 1) * (1 / 12)) / (n0 * n1)
    se_ind = var_ind ** 0.5
    mde_ind = 0.5 + (Z975 + 0.841621234) * se_ind
    print("MDE analysis -- pixel-domain rerun audit (n=200: 50 clips x [1 H0 + 3 H1])")
    print(f"independence floor: SE = {se_ind:.5f}  ->  MDE AUC = {mde_ind:.4f}"
          "  (alpha=0.05 two-sided, power 0.80)")
    print()
    hdr = (f"{'stage':<9}{'AUC_obs':>9}{'ICC(1)':>8}{'SE_boot':>9}{'DEFF':>7}"
           f"{'MDE d':>9}{'MDE AUC':>9}{'power@d':>9}")
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for st in STAGES:
        X = stage_matrix(rows, clips, st)
        if np.allclose(X, 0.0):
            print(f"{st:<9}{'--':>9}  (degenerate, excluded)")
            continue
        obs = auc(X.reshape(-1), LABELS)
        se = boot_null_auc(X, rng, B_SE).std(ddof=1)
        deff = (se / se_ind) ** 2
        crit = 0.5 + Z975 * se
        d = mde_by_bisection(X, crit, rng)
        pw = power_at(X, d, crit, rng)
        print(f"{st:<9}{obs:>9.4f}{icc1_h1_arm(X):>8.3f}{se:>9.5f}{deff:>7.2f}"
              f"{d:>9.4f}{population_auc(X, d):>9.4f}{pw:>9.3f}")
        results[st] = (obs, se, deff, d, population_auc(X, d))
    print()
    print("SE_boot: std of the AUC over B=%d clip-bootstrap replicates, each realising"
          % B_SE)
    print("  H0 by choosing 1 of 4 rows per resampled clip as pseudo-H0 (seed %d)."
          % SEED)
    print("DEFF = (SE_boot/SE_indep)^2.  MDE d: shift added to all 3 H1 rows such that")
    print("  clip-bootstrap power (B=%d) reaches 0.80; MDE AUC = P(S0 < S1+d) on the" % B_POWER)
    print("  full observed sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
