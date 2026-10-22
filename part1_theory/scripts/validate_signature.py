"""Numerical validation of Section IV results (signature existence).

Targets
-------
Theorem 1  (double-quantization periodicity)   h[k + q2/g] = h[k]
Corollary 1 (empty-bin pattern)                occupied fraction = g/q2
Lemma 1    (GGD histogram under single quant.)  exact mass formula vs empirical
Lemma 2    (GGD histogram under double quant.)  exact mass formula vs empirical
Proposition 4 (periodicity strength)           comb-replication identity and the
                                               actual scaling of the DFT peak

The last item is diagnostic: the draft asserts P(h) >= c/Delta = c*g/q2, i.e. the
fundamental carries a Theta(1/Delta) share.  We test that directly.

Run:  python scripts/theory/validate_signature.py
Output: data/signature_validation.txt   and   figs/theorem1_comb_replication.png
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.special import gammaln


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

SEED = 20261005


# ---------------------------------------------------------------- GGD machinery
def ggd_cdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """CDF of the zero-mean generalized Gaussian with scale alpha, shape beta."""
    u = np.abs(x / alpha)
    # For 0 < p < 1 of lower-incomplete-gamma we use the regularized form via
    # scipy's gammainc imported lazily.
    from scipy.special import gammainc

    a = 1.0 / beta
    return 0.5 + 0.5 * np.sign(x) * gammainc(a, u**beta)


def ggd_sample(rng: np.random.Generator, n: int, alpha: float, beta: float) -> np.ndarray:
    """Sample a zero-mean GGD as  S * |Z|^(1/beta)  with Z ~ GeneralizedGamma.

    Uses the standard representation: if G ~ Gamma(shape=1/beta, scale=1) then
    X = alpha * G^(1/beta) has density proportional to exp(-(|x|/alpha)^beta)
    on the half line; the random sign restores symmetry.
    """
    shape = 1.0 / beta
    g = rng.gamma(shape, 1.0, size=n)
    x = alpha * np.power(g, 1.0 / beta)
    return x * rng.choice([-1.0, 1.0], size=n)


def quantize(x: np.ndarray, q: float) -> np.ndarray:
    """Uniform mid-tread quantizer Q_q(x) = q * round(x / q) with ties upward."""
    return q * np.floor(x / q + 0.5)


# ------------------------------------------------------------------- histograms
def build_histogram(values: np.ndarray, n_bins: int, width: float) -> np.ndarray:
    """Symmetric histogram with n_bins cells of given `width`, centred at 0.

    Requires n_bins even so that the origin is a cell boundary; the mapping is
    k(v) = floor(v / width) + n_bins // 2.
    """
    assert n_bins % 2 == 0, "n_bins must be even"
    idx = np.floor(values / width).astype(np.int64) + n_bins // 2
    idx = np.clip(idx, 0, n_bins - 1)
    h = np.bincount(idx, minlength=n_bins).astype(float)
    return h


def bin_of_value(v, width, n_bins):
    return int(math.floor(v / width)) + n_bins // 2


# ------------------------------------------------------------------- DFT scores
def periodicity_score(h: np.ndarray) -> float:
    """P(h) = max_{l != 0} |hat h(l)| / mean_l |hat h(l)|   (draft Eq. for Prop 4)."""
    hat = np.fft.fft(h)
    mag = np.abs(hat)
    return float(mag[1:].max() / mag.mean())


def _fmt(x, nd=6):
    return ("%." + str(nd) + "f") % x


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit("=" * 78)
    emit("Section IV validation: double-quantization periodicity")
    emit("=" * 78)

    # ---------------------------------------------------------- parameters ----
    alpha, beta = 12.0, 1.0          # Laplace-tailed GGD: light enough that no coefficient
                               # falls outside the histogram window (assumption A1)
    N = 2_000_000                    # Monte-Carlo samples

    # We need the histogram length L to be an integer multiple of the comb
    # period b = q2/g so that the replication identity is testable exactly.
    cases = [(4, 8), (4, 12), (6, 9), (8, 12), (2, 6)]

    X = ggd_sample(rng, N, alpha, beta)

    # ------------------------------------------------ Lemma 1 & 2 (masses) ----
    emit("")
    emit("[Lemma 1/2] closed-form masses vs Monte-Carlo")
    emit("  q1    q2    max| Lem1 err |   max| Lem2 err |")
    lem_ok = True
    for q1, q2 in cases:
        Y1 = quantize(X, q1)
        Y2 = quantize(Y1, q2)

        # --- Lemma 1 ---
        ks = np.arange(-6, 7)
        grid = np.array([(k - 0.5) * q1 for k in ks] + [(k + 0.5) * q1 for k in ks])
        F = ggd_cdf(grid, alpha, beta)
        nk = len(ks)
        Flo, Fhi = F[:nk], F[nk:]
        pred1 = Fhi - Flo
        emp1 = np.array([np.mean(Y1 == k * q1) for k in ks])
        err1 = float(np.abs(pred1 - emp1).max())

        # --- Lemma 2 ---
        ms = np.arange(-3, 4)
        pred2 = []
        for m in ms:
            lo, hi = (m - 0.5) * q2, (m + 0.5) * q2
            ns = np.arange(math.ceil(lo / q1), math.ceil(hi / q1))
            tot = 0.0
            for nn in ns:
                tot += ggd_cdf(np.array([(nn + 0.5) * q1]), alpha, beta)[0] - \
                       ggd_cdf(np.array([(nn - 0.5) * q1]), alpha, beta)[0]
            pred2.append(tot)
        pred2 = np.array(pred2)
        emp2 = np.array([np.mean(Y2 == m * q2) for m in ms])
        err2 = float(np.abs(pred2 - emp2).max())
        lem_ok &= (err1 < 5e-3) and (err2 < 5e-3)
        emit("  %-5d %-5d  %-14s   %-14s" % (q1, q2, _fmt(err1, 6), _fmt(err2, 6)))
    emit("  verdict: %s" % ("PASS" if lem_ok else "FAIL"))

    # ------------------------------------- Theorem 1 & Corollary 1 (comb) -----
    emit("")
    emit("[Theorem 1 / Corollary 1] comb period and occupied fraction")
    emit("  q1    q2    g   Delta   period-match    occupied(frac)  theory g/q2")
    comb_ok = True
    for q1, q2 in cases:
        g = math.gcd(int(q1), int(q2))
        b = int(q2) // g                    # Delta
        L = 2 * b * 512                     # even, multiple of b, and wide enough for A1
        Y2 = quantize(quantize(X, q1), q2)
        h = build_histogram(Y2, L, g)

        # period check: compare h[k] with h[k+b] WITHOUT wraparound (the wraparound
        # of np.roll would compare the last b bins against the first b bins, which
        # are not related by the theorem)
        frac_out = float(np.abs(h[:-b] - h[b:]).sum() / max(h.sum(), 1e-12))

        occ = float((h > 0).sum() / L)
        theory = g / q2
        ok = (frac_out < 1e-12) and abs(occ - theory) < 0.02
        comb_ok &= ok
        emit("  %-5d %-5d %-3d %-6d  %.2e     %-14s  %-8s  %s"
             % (q1, q2, g, b, frac_out, _fmt(occ, 4), _fmt(theory, 4),
                "PASS" if ok else "FAIL"))
    emit("  verdict: %s" % ("PASS" if comb_ok else "FAIL"))

    # -------------------------------- Proposition 4 (comb replication) --------
    emit("")
    emit("[Proposition 4] comb-replication identity and DFT peak scaling")
    emit("  Identity under test:  hat h(l) = hat w(l mod M),  w_m = h[m*b], L = M*b")
    emit("  q1    q2    g   Delta   max|hat h - rep|/N   P(h)    |hat h(M)|/N")
    rep_rows = []
    for q1, q2 in cases:
        g = math.gcd(int(q1), int(q2))
        b = int(q2) // g
        M = 64
        L = M * b
        Y2 = quantize(quantize(X, q1), q2)
        h = build_histogram(Y2, L, g)
        Nn = h.sum()

        hat = np.fft.fft(h)
        w = h.reshape(M, b)[:, 0]                # occupied bins are k = 0 mod b
        w_hat = np.fft.fft(w)

        # replication of w_hat b times
        rep = np.concatenate([w_hat] * b)
        err = float(np.abs(hat - rep).max() / max(Nn, 1e-12))

        P = periodicity_score(h)
        fund = float(np.abs(hat[M]) / Nn)        # l = M  <=>  omega = 2*pi/Delta
        rep_rows.append((q1, q2, g, b, err, P, fund))
        emit("  %-5d %-5d %-3d %-6d  %-16s  %-8s %-8s"
             % (q1, q2, g, b, _fmt(err, 8), _fmt(P, 4), _fmt(fund, 4)))

    rep_ok = max(r[4] for r in rep_rows) < 1e-10
    emit("  replication identity: %s" % ("PASS" if rep_ok else "FAIL"))

    # does the fundamental decay as 1/Delta ?
    emit("")
    emit("  Fundamental magnitude |hat h(omega_Delta)| / N versus Delta:")
    emit("     draft Claim (Prop 4 / Remark): Theta(1/Delta)")
    for r in rep_rows:
        emit("       Delta = %-4d   measured = %-8s   1/Delta = %-8s"
             % (r[3], _fmt(r[6], 4), _fmt(1.0 / r[3], 4)))

    emit("")
    emit("[DIAGNOSIS]")
    funds = [r[6] for r in rep_rows]
    if min(funds) > 0.9:
        emit("  The fundamental does NOT decay as 1/Delta: it equals N to within a")
        emit("  few percent for every lattice tested.  Under the exact comb model")
        emit("  hat h(M) = hat w(0) = N, i.e. the comb replicates the DC term onto")
        emit("  the fundamental index, so the Theta(1/Delta) claim of Prop 4 and the")
        emit("  Theta(eps) claim of Prop 7 are NOT consequences of Theorem 1 alone.")
        emit("  The two statements require an additional assumption, stated in the")
        emit("  revised proofs: the comb must be read against a *within-bin*")
        emit("  reference, i.e. the idle-tone model below.")
    emit("=" * 78)

    txt = os.path.join(DATA, "signature_validation.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\nwrote " + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
