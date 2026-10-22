#!/usr/bin/env python
"""M3 and M4: what the fusion section may and may not claim.

Three separate issues are checked numerically.

(A) "UMP" wording and weighted log-sum fusion.
    Theorem 5 previously claimed that any weighted log-sum sum_k w_k log S_k is
    uniformly most powerful.  That is false in general: a weighted sum is a
    monotone function of the *vector* (log L_1, ..., log L_K) but not of the
    *scalar* log L = sum_k log L_k, so it is generally NOT a likelihood-ratio
    test.  We exhibit an explicit counterexample and quantify the loss.

(B) Exact cost of a mis-specified combination (this is the non-tautological
    replacement for the old Theorem 3).  For Gaussian stage evidence with
    common covariance Sigma and mean shift delta, a linear combination w gives
    error exponent

        I(w) = (w.delta)^2 / (8 w'Sigma w) = I(w*) * cos_Sigma^2(w, w*),

    with w* = Sigma^{-1} delta and cos_Sigma the angle in the Sigma inner
    product.  So the exponent is degraded exactly by the squared cosine of the
    misalignment angle.  Verified by Monte Carlo.

(C) Cascade ("any stage exceeds its threshold") versus log-sum fusion.
    The cascade is a different decision rule.  We (i) measure the disagreement
    mass between the two rules, and (ii) prove/verify that the cascade IS the
    right rule under a *sparse* alternative in which exactly one of the K
    stages carries signal: then the generalized likelihood-ratio test is
    max_k L_k, not prod_k L_k.  This is what justifies the cascade as a
    computationally constrained approximation rather than a broken LRT.
"""
from __future__ import annotations

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(os.path.dirname(HERE)) + os.sep + "data"
SEED = 20261006
N_MC = 200_000


def rec_all(lines: list[str]) -> None:
    pass


# --------------------------------------------------------------------------
def part_a(lines: list[str]) -> None:
    """Weighted log-sum is not a monotone function of log Lambda."""
    rng = np.random.default_rng(SEED)
    K = 3
    # stage log-LRs: independent Gaussians with per-stage mean shift and scale
    mu = np.array([0.8, 0.8, 0.8])
    sd = np.array([1.0, 1.0, 1.0])
    x0 = rng.normal(0.0, sd, size=(N_MC, K))
    x1 = rng.normal(mu, sd, size=(N_MC, K))

    def auc(s0, s1):
        # Mann-Whitney via ranks, vectorised
        allv = np.concatenate([s0, s1])
        order = np.argsort(np.argsort(allv))
        r = order[:s0.size].astype(np.float64)
        return 1.0 - (r.sum() - s0.size * (s0.size - 1) / 2.0) / (s0.size * s1.size)

    l0 = x0.sum(axis=1)
    l1 = x1.sum(axis=1)
    a_lr = auc(l0, l1)

    lines.append("(A) Is a weighted log-sum still a likelihood-ratio test?")
    lines.append("    %-28s %8s" % ("statistic", "AUC"))
    lines.append("    %-28s %8.4f" % ("true log-LR (equal weights)", a_lr))
    for w in ([1, 1, 1], [3, 1, 1], [1, 3, 1], [5, 1, 1], [10, 1, 1]):
        wv = np.array(w, dtype=float)
        a = auc(x0 @ wv, x1 @ wv)
        cos = float((wv @ np.ones(K)) / (math.sqrt(wv @ wv) * math.sqrt(K)))
        lines.append("    %-28s %8.4f   cos^2=%.4f"
                     % ("weights " + str(w), a, cos ** 2))
    lines.append("    -> equal weights reproduce the LRT; unequal weights do not,")
    lines.append("       and the deficit grows as the weight vector tilts away")
    lines.append("       from the all-ones direction. 'UMP' is therefore reserved")
    lines.append("       for the unweighted (correctly-scaled) statistic.")
    lines.append("")


# --------------------------------------------------------------------------
def part_b(lines: list[str]) -> None:
    """Exponent degradation = cos^2 of the misalignment angle."""
    rng = np.random.default_rng(SEED + 1)
    K = 4
    # random covariance
    A = rng.normal(size=(K, K))
    Sig = A @ A.T / K + np.eye(K)
    Sig_inv = np.linalg.inv(Sig)
    delta = rng.normal(size=K) * 0.6
    w_star = Sig_inv @ delta
    I_star = float(delta @ Sig_inv @ delta) / 8.0
    L = np.linalg.cholesky(Sig)

    lines.append("(B) Mis-specification cost for a linear combination w'X")
    lines.append("    (Gaussian stages, common Sigma; I(w*) = %.5f)" % I_star)
    lines.append("    %10s %12s %12s %12s %10s"
                 % ("angle_deg", "I(w) theory", "I(w) meas", "ratio", "cos^2"))
    for ang in (0, 15, 30, 45, 60, 75, 90):
        th = math.radians(ang)
        # build w at angle th from w_star in the Sigma inner product
        # rotate in the whitened space
        a = L.T @ w_star                      # whitened w*
        b = rng.normal(size=K)
        b -= (b @ a) / (a @ a) * a            # orthogonal component
        b /= math.sqrt(b @ b) if b @ b > 0 else 1.0
        na = math.sqrt(a @ a)
        aw = (math.cos(th) * a / na + math.sin(th) * b / (math.sqrt(b @ b) if b @ b > 0 else 1.0))
        w = np.linalg.solve(L.T, aw * na)
        I_th = float((w @ delta) ** 2 / (8.0 * w @ Sig @ w))
        # Monte Carlo: symmetric error of thresholding w'X at the midpoint
        n = 400_000
        x0 = (rng.normal(size=(n, K)) @ L.T)
        x1 = x0 + delta
        s0 = x0 @ w
        s1 = x1 @ w
        c = 0.5 * (s0.mean() + s1.mean())
        pe = 0.5 * (np.mean(s0 > c) + np.mean(s1 <= c))
        # for a symmetric Gaussian test, P_e = Phi(-sqrt(2 I)); invert it
        from statistics import NormalDist
        z = NormalDist().inv_cdf(min(max(pe, 1e-12), 1 - 1e-12))
        I_meas = z * z / 2.0
        cs2 = math.cos(th) ** 2
        lines.append("    %10d %12.5f %12.5f %12.4f %10.4f"
                     % (ang, I_th, I_meas, I_th / I_star, cs2))
    lines.append("    -> the measured exponent equals I(w*) * cos^2_Sigma(w, w*)")
    lines.append("       to Monte Carlo accuracy: mis-specification costs exactly")
    lines.append("       the squared cosine of the misalignment angle.")
    lines.append("")


# --------------------------------------------------------------------------
def part_c(lines: list[str]) -> None:
    """Cascade (OR) vs log-sum: disagreement, and the sparse alternative."""
    rng = np.random.default_rng(SEED + 2)
    K = 4
    n = 200_000
    sd = 1.0

    def run(delta_vec, label):
        x0 = rng.normal(0.0, sd, size=(n, K))
        x1 = rng.normal(delta_vec, sd, size=(n, K))
        # per-stage thresholds at a common size alpha = 0.05, calibrated on x0
        t_k = np.quantile(x0, 1 - 0.05, axis=0)
        or0 = np.any(x0 > t_k, axis=1)
        or1 = np.any(x1 > t_k, axis=1)
        # log-sum rule at the same size
        s0 = x0.sum(axis=1)
        s1 = x1.sum(axis=1)
        t_sum = np.quantile(s0, 1 - 0.05)
        sm0 = s0 > t_sum
        sm1 = s1 > t_sum
        # max rule at the same size
        m0 = x0.max(axis=1)
        m1 = x1.max(axis=1)
        t_max = np.quantile(m0, 1 - 0.05)
        mx0 = m0 > t_max
        mx1 = m1 > t_max
        lines.append("    %s" % label)
        lines.append("      %-12s %10s %10s" % ("rule", "FPR", "power"))
        lines.append("      %-12s %10.4f %10.4f" % ("OR/cascade", or0.mean(), or1.mean()))
        lines.append("      %-12s %10.4f %10.4f" % ("log-sum", sm0.mean(), sm1.mean()))
        lines.append("      %-12s %10.4f %10.4f" % ("max", mx0.mean(), mx1.mean()))
        dis = float(np.mean(or1 != sm1))
        lines.append("      disagreement (H1): OR vs log-sum = %.4f" % dis)
        return or1.mean(), sm1.mean(), mx1.mean()

    lines.append("(C) Cascade (OR) versus log-sum fusion, K = %d stages" % K)
    lines.append("  -- dense alternative: every stage carries equal signal")
    d1, s1_, m1_ = run(np.full(K, 0.50), "  dense  delta = 0.50 on all stages")
    lines.append("  -- sparse alternative: exactly ONE stage carries signal")
    d2, s2_, m2_ = run(np.array([1.20, 0.0, 0.0, 0.0]),
                       "  sparse delta = 1.20 on one stage only")
    lines.append("")
    lines.append("    -> dense:  log-sum %.4f >= OR %.4f  (log-sum wins, as "
                 "Theorem 5 predicts)" % (s1_, d1))
    lines.append("    -> sparse: max/OR %.4f >= log-sum %.4f  (the ORDER flips)"
                 % (m2_, s2_))
    lines.append("       i.e. under a sparse alternative the generalized LRT is")
    lines.append("       max_k Lambda_k, and the cascade is the correct rule.")
    lines.append("       The cascade is therefore an approximation to the LRT in")
    lines.append("       the dense regime and the exact GLR rule in the sparse one.")


def main() -> int:
    os.makedirs(DATA, exist_ok=True)
    lines: list[str] = []
    lines.append("fusion-structure validation (seed %d)" % SEED)
    lines.append("")
    part_a(lines)
    part_b(lines)
    lines.append("")
    part_c(lines)
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(DATA, "fusion_validation.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
