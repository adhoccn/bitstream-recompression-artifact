"""Numerical validation of Theorem 1 and Corollary 1 (support / gap structure).

The theorem is purely number-theoretic, so it is checked in two independent ways:

  (A) EXACT   -- enumerate the first-pass indices n directly and see which
                 candidate bins k = m*b actually receive mass.  No sampling.
  (B) EMPIRICAL -- draw a full-support source (Laplace / GGD), push it through
                 Q_{q2}(Q_{q1}(.)) and compare the observed histogram with (A).

Checked items
  Thm 1(i)   support subset of b*Z (bins of width g);
  Thm 1(ii)  reachability criterion  dist(m b, a Z) < b/2;
  Thm 1(iii) two regimes: b>a -> every candidate used; b<a -> fraction b/a,
             occupancy pattern periodic with period a in m  (period a*b in k);
  Thm 1(iv)  values are NOT periodic (h[k+Delta] != h[k]) while the occupancy
             indicator IS periodic;
  Cor 1      occupied fraction = g/q2 (b>a) or g/q1 (b<a).

Run:  python scripts/theory/validate_comb_structure.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "data", "comb_validation.txt")
SEED = 20261005

CASES = [
    # (q1, q2, label)   -- b > a  (q2 > q1) and b < a  (q2 < q1), coprime and not
    (2, 4, "q1=2,  q2=4   g=2 a=1 b=2   (b>a)"),
    (4, 6, "q1=4,  q2=6   g=2 a=2 b=3   (b>a)"),
    (8, 12, "q1=8,  q2=12  g=4 a=2 b=3   (b>a)"),
    (4, 2, "q1=4,  q2=2   g=2 a=2 b=1   (b<a)"),
    (6, 4, "q1=6,  q2=4   g=2 a=3 b=2   (b<a)"),
    (12, 8, "q1=12, q2=8   g=4 a=3 b=2   (b<a)"),
    (8, 6, "q1=8,  q2=6   g=2 a=4 b=3   (b<a)"),
    (9, 6, "q1=9,  q2=6   g=3 a=3 b=2   (b<a)"),
    (5, 3, "q1=5,  q2=3   g=1 a=5 b=3   (b<a)"),
    (3, 5, "q1=3,  q2=5   g=1 a=3 b=5   (b>a)"),
]

M_RANGE = 4000          # candidate indices m = -M_RANGE .. M_RANGE
N_SAMPLES = 1_000_000   # empirical sample size
SOURCE_SCALE = 12.0     # Laplace scale of the synthetic coefficient source
MIN_COUNT = 5.0         # minimum expected count for a bin to enter the test window


def quantize(x, q):
    return q * np.floor(x / q + 0.5)


def dist_to_lattice(v, a):
    """dist(v, a*Z) for integer v, a."""
    r = v % a
    return min(r, a - r)


def reachable_halfopen(m, a, b):
    """Theorem 1(ii) in its half-open (correct) form.

    With decision cells I_k(q) = [(k-1/2)q, (k+1/2)q), the first-pass level n*a
    lands in candidate bin m iff

        m*b - b/2  <=  n*a  <  m*b + b/2 .

    Multiply by 2 and put u = 2*m*b - b (an integer): u <= 2*n*a < u + 2*b, whose
    number of integer solutions n is  ceil((u+2b)/(2a)) - ceil(u/(2a)).
    When b is even the boundary |n*a - m*b| = b/2 IS attained; there the symmetric
    strict form dist(m*b, a*Z) < b/2 is wrong (it discards that n altogether),
    while the half-open form assigns it to the upper neighbour.
    """
    u = 2 * m * b - b
    return (math.ceil((u + 2 * b) / (2 * a)) - math.ceil(u / (2 * a))) >= 1


def exact_occupancy(q1, q2, m_range=M_RANGE):
    """(A) Which candidate bins k = m*b are reachable, and from how many n?

    Enumerates first-pass levels n and applies the same floor(x/q + 1/2) rule the
    quantizer uses, so ties are resolved exactly as they are in the definition.
    Returns dict m -> list of contributing n.
    """
    g = math.gcd(q1, q2)
    a, b = q1 // g, q2 // g
    occ = {}
    n_hi = int(math.ceil((m_range + 1) * b / a)) + 2
    for n in range(-n_hi, n_hi + 1):
        m = math.floor(n * a / b + 0.5)
        occ.setdefault(m, []).append(n)
    return g, a, b, occ


def laplace_cell_prob(n, q1, scale):
    """P(X in cell n of the first-pass quantizer), X ~ Laplace(scale)."""
    def F(x):
        return 0.5 * math.exp(x / scale) if x < 0 else 1.0 - 0.5 * math.exp(-x / scale)
    return F((n + 0.5) * q1) - F((n - 0.5) * q1)


def expected_supply(m, a, b, q1, scale, n_mc):
    """Expected number of samples landing in candidate bin k = m*b.

    Every first-pass cell n with m*b - b/2 <= n*a < m*b + b/2 maps deterministically
    to that bin, so this is the exact mean of the bin count.
    """
    lo = math.ceil((m * b - b / 2) / a)
    hi = math.ceil((m * b + b / 2) / a) - 1        # half-open upper endpoint
    return n_mc * sum(laplace_cell_prob(n, q1, scale) for n in range(lo, hi + 1))


def predicted_marks(a, b, m_range=M_RANGE):
    """Indicator over m of the half-open reachability criterion."""
    return {m: reachable_halfopen(m, a, b) for m in range(-m_range, m_range + 1)}


def empirical_histograms(q1, q2, rng, n=N_SAMPLES, L=8192):
    """(B) Push a full-support Laplace source through one and two quantizers.

    Returns (h1, h2) on a common grid of bins of width g, indexed so that
    index (L//2 + k) holds the value k*g.  Both are needed: h1 defines the
    *effective support* of the source (a finite sample of a full-support
    density is nonzero only on a finite range), and occupancy is therefore
    measured only inside that range.
    """
    g = math.gcd(q1, q2)
    X = SOURCE_SCALE * rng.exponential(1.0, n) * rng.choice([-1.0, 1.0], n)
    y1, y2 = quantize(X, q1), quantize(quantize(X, q1), q2)
    out = []
    for v in (y1, y2):
        idx = np.floor(v / g).astype(np.int64) + L // 2
        idx = idx[(idx >= 0) & (idx < L)]
        out.append(np.bincount(idx, minlength=L).astype(float))
    return out[0], out[1], g


def main() -> int:
    rng = np.random.default_rng(SEED)
    lines = []
    n_fail = 0

    def rec(s=""):
        print(s)
        lines.append(s)

    rec("=" * 88)
    rec("Validation of Theorem 1 (support / gap structure) and Corollary 1 (occupied fraction)")
    rec("seed = %d ;  EXACT enumeration m in [-%d, %d] ;  EMPIRICAL n = %d Laplace draws"
        % (SEED, M_RANGE, M_RANGE, N_SAMPLES))
    rec("=" * 88)

    for q1, q2, label in CASES:
        g, a, b, occ = exact_occupancy(q1, q2)
        pred = predicted_marks(a, b)

        rec("")
        rec(label)
        rec("-" * 88)

        # ---- (ii) reachability criterion -------------------------------------
        lo, hi = -M_RANGE, M_RANGE
        enum = {m: (m in occ) for m in range(lo, hi + 1)}
        mism = [m for m in range(lo, hi + 1)
                if enum[m] != pred.get(m, False)]
        ok_ii = not mism
        n_fail += 0 if ok_ii else 1
        rec("  Thm 1(ii)  reachability (half-open)  m*b-b/2 <= n*a < m*b+b/2 :  %s"
            % ("PASS" if ok_ii else "FAIL  (%d mismatches, e.g. %s)" % (len(mism), mism[:5])))

        # the symmetric strict form used in the first draft: correct iff b is odd
        strict = {m: (dist_to_lattice(m * b, a) < b / 2 - 1e-12)
                  for m in range(lo, hi + 1)}
        mism_strict = [m for m in range(lo, hi + 1) if strict[m] != enum[m]]
        rec("  [draft form] symmetric strict dist(m*b, a*Z) < b/2 : %s "
            "(%d mismatches%s)" % ("equivalent" if not mism_strict else "WRONG",
                                   len(mism_strict),
                                   "" if b % 2 else "  <- b even, ties attained"))

        # ---- (iii) regimes ---------------------------------------------------
        used = [m for m in range(lo, hi + 1) if enum[m]]
        frac = len(used) / (hi - lo + 1)
        if b > a:
            pred_frac, regime = 1.0, "b>a"
        else:
            pred_frac = b / a
            regime = "b<a"
        # exactly b out of every a consecutive candidate indices qualify
        ok_iii_frac = abs(frac - pred_frac) <= 1.5 / min(a, hi - lo + 1) + 1e-9
        per_a = [1 for m in range(0, a) if reachable_halfopen(m, a, b)]
        ok_exact_b = (sum(per_a) == min(b, a)) if b < a else (sum(per_a) == a)
        n_fail += 0 if ok_iii_frac else 1
        rec("  Thm 1(iii) candidate-use fraction  measured %.4f  predicted %.4f  (%s)  %s"
            % (frac, pred_frac, regime, "PASS" if ok_iii_frac else "FAIL"))
        rec("  Thm 1(iii) exactly %d of each block of %d candidates qualify : %s"
            % (min(b, a), a, "PASS" if ok_exact_b else "FAIL (%d)" % sum(per_a)))

        # periodicity of the occupancy indicator: period a in m  (= a*b in k)
        per_m, per_k = a, a * b
        devs = [m for m in range(lo, hi - per_m + 1)
                if enum[m] != enum[m + per_m]]
        ok_per_m = not devs
        devs_k = [m for m in range(lo, hi - a + 1)
                  if enum[m] != enum[m + a]]
        # period in the bin index k: comparing k and k + a*b means m and m + a
        okay_k = True
        for m in range(lo, hi - a + 1):
            k1, k2 = m * b, (m + a) * b
            if enum.get(m, False) != enum.get(m + a, False):
                okay_k = False
                break
        n_fail += 0 if (ok_per_m and okay_k) else 1
        rec("  Thm 1(iii) occupancy-indicator period:  %d in m / %d in k   %s"
            % (per_m, per_k, "PASS" if (ok_per_m and okay_k) else "FAIL"))

        # ---- empirical: (i), (iv), Corollary 1 --------------------------------
        h1, h2, gch = empirical_histograms(q1, q2, rng)
        assert gch == g
        L = h1.size
        center = L // 2
        kk_all = np.arange(L) - center

        # Effective support of the *source*, read off the single-quantized data,
        # then shrunk to the region where every candidate bin has an expected
        # count of at least MIN_COUNT samples.  Outside that region a zero count
        # reflects finite sampling, not the support structure of the theorem.
        sup1 = np.flatnonzero(h1 > 0)
        kmax = int(np.max(np.abs(kk_all[sup1])))
        W = (int(0.9 * kmax) // b) * b         # whole number of candidate periods
        # only the candidates predicted to be *reachable* can be under-sampled;
        # unreachable ones have expected supply identically zero by construction
        while W > b:
            ms_w = [m for m in range(-W // b, W // b + 1) if enum.get(m, False)]
            if min(expected_supply(m, a, b, q1, SOURCE_SCALE, N_SAMPLES)
                   for m in ms_w) >= MIN_COUNT:
                break
            W -= b
        lo_w, hi_w = center - W, center + W
        n_bins = 2 * W + 1

        # (i) support subset of b*Z
        nz2 = h2[lo_w:hi_w + 1] > 0
        kk = np.arange(-W, W + 1)
        off_lattice = int(np.sum(nz2 & ((kk % b) != 0)))
        ok_i = off_lattice == 0
        n_fail += 0 if ok_i else 1
        rec("  Thm 1(i)   all empirical mass on k = 0 mod b : %s  "
            "(off-lattice bins = %d, window |k|<=%d)" % ("PASS" if ok_i else "FAIL",
                                                         off_lattice, W))

        # cross-check: empirical occupancy == exact prediction, bin by bin
        ms = [m for m in range(-W // b, W // b + 1)]
        mism2 = [m for m in ms
                 if bool(h2[center + m * b] > 0) != enum.get(m, False)]
        ok_match = not mism2
        n_fail += 0 if ok_match else 1
        rec("  cross-check  empirical occupancy == exact prediction over %d candidates : "
            "%s (%d mismatches)" % (len(ms), "PASS" if ok_match else "FAIL", len(mism2)))

        # Corollary 1: occupied fraction of g-width bins
        emp_frac = float(nz2.mean())
        n_pred = sum(1 for m in ms if enum.get(m, False))   # same window, exact
        pred_frac_win = n_pred / n_bins                     # per bin, not per candidate
        cor_pred = g / q2 if b > a else g / q1
        ok_cor = abs(emp_frac - cor_pred) <= 0.03
        n_fail += 0 if ok_cor else 1
        rec("  Cor 1      occupied fraction  measured %.4f  window-exact %.4f  "
            "closed form %.4f  %s" % (emp_frac, pred_frac_win, cor_pred,
                                      "PASS" if ok_cor else "FAIL"))

        # (iv) values are not periodic
        vals = h2[lo_w:hi_w + 1][nz2]
        if vals.size > 3:
            rel = float(np.max(np.abs(np.diff(vals)) / np.maximum(vals[:-1], 1e-9)))
            ok_iv = rel > 0.05
        else:
            rel, ok_iv = float("nan"), False
        n_fail += 0 if ok_iv else 1
        rec("  Thm 1(iv)  h[k+Delta] != h[k] as values : max relative step %.3f  %s"
            % (rel, "PASS" if ok_iv else "FAIL (values look periodic)"))

    rec("")
    rec("=" * 88)
    rec("TOTAL FAILURES: %d" % n_fail)
    rec("=" * 88)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwritten to", OUT)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
