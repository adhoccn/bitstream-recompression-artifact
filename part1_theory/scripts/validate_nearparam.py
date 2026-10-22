"""Numerical validation of the near-parameter detectability bounds (Section VII).

Chain that replaces the draft's FFT-SNR heuristic:

  Step 1   TV(P_D, P_2) = Theta(eps)          Proposition 7 (revised)
  Step 2   H^2(P_D, P_2) = Theta(eps^2)       Lemma (Hellinger vs smooth perturbation)
  Step 3   Le Cam: err_0 + err_1 >= 1 - d_TV(P^N, Q^N) >= 1 - 2*sqrt(N)*H(P,Q)
           =>  N >= (1-2*delta)^2 / (4 H^2)  =  Omega(eps^{-2})    Theorem 6

We validate every link numerically:
  * measure TV, H^2, chi^2 against eps and fit power-law exponents;
  * run an ACTUAL Neyman-Pearson test on finite samples and record the minimum
    sample size N_min(eps) at which (miss, FA) <= delta, then fit its exponent.

Run: python scripts/theory/validate_nearparam.py [--seeds N]

Replication (added 2026-10-06, reviewer request): the fitted power-law
exponents of Step 1+2 are reported as mean +/- sd over N independent seeds
(N_SEEDS below, override with --seeds).  Seed SEED+0 reproduces the original
single-sample run exactly -- it is the first draw from default_rng(SEED) -- so
the plotted curve and the headline numbers are unchanged; seeds SEED+1.. are
additional independent replications and cost one 4e6 GGD draw each.

Step 3 (the empirical NP test) stays single-seed: it is ~O(10) more expensive
and is not the quantity quoted with an exponent in the paper.  This is stated
in the output.

Outputs: data/nearparam_validation.txt
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(DATA, exist_ok=True)

SEED = 20261005
N_SEEDS = 10


def ggd_sample(rng, n, alpha, beta):
    g = rng.gamma(1.0 / beta, 1.0, size=n)
    return alpha * np.power(g, 1.0 / beta) * rng.choice([-1.0, 1.0], size=n)


def quantize(x, q):
    return q * np.floor(x / q + 0.5)


def pmf_on_lattice(values, q, width=6001):
    """Empirical pmf of `values` on cells { (k - width//2) * q }."""
    c = np.floor(values / q + 0.5).astype(np.int64)
    c = np.clip(c, -(width // 2), width // 2)
    z = np.bincount(c + width // 2, minlength=width).astype(float)
    return z / z.sum()


def divergences(p, q):
    tv = 0.5 * np.abs(p - q).sum()
    h2 = float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    m = (p + q) / 2.0
    nz = m > 1e-15
    chi2 = float(np.sum((p[nz] - q[nz]) ** 2 / m[nz]))
    return tv, h2, chi2


# (q1, target) sweep of Step 1+2, shared by the main run and the replications.
# SWEEP_Q1 matches fig_nearparam_synthetic() in make_figures.py: the fitted
# exponents quoted in the Fig. 5 caption come from that grid (18 pairs), and an
# earlier q1 = 8 in this script silently produced different exponents (+1.000
# against the figure's +0.941) for the same chain.  Keep the two in step.
SWEEP_Q1 = (16, 32, 64)
SWEEP_TARGET = (0.40, 0.25, 0.15, 0.08, 0.04, 0.02)


def sweep_rows(X):
    """Measure (q1, q2, eps, TV, H^2, chi^2) over the fixed (q1, target) grid."""
    rows = []
    for q1 in SWEEP_Q1:
        for target in SWEEP_TARGET:
            q2 = max(q1 + 1, int(math.ceil(q1 * (1 + target))))
            if math.gcd(int(q1), int(q2)) != 1:
                q2 += 1
            eps = q2 / q1 - 1.0
            if eps <= 0 or eps > 0.5:
                continue
            PD = pmf_on_lattice(quantize(quantize(X, q1), q2), q2)
            P2 = pmf_on_lattice(quantize(X, q2), q2)
            tv, h2, chi2 = divergences(PD, P2)
            rows.append((q1, q2, eps, tv, h2, chi2))
    return rows


def fit_exponents(rows):
    """Power-law exponent of each divergence against eps, from log-log fit."""
    eps_a = np.array([r[2] for r in rows])
    exps = {}
    for idx, name in ((3, "TV"), (4, "H^2"), (5, "chi^2")):
        v = np.array([r[idx] for r in rows])
        p = np.polyfit(np.log(eps_a), np.log(v), 1)
        exps[name] = float(p[0])
    return exps


def replicate_exponents(seed, alpha, beta, N_MC):
    """One independent replication: fresh draw, same grid, fitted exponents."""
    rng = np.random.default_rng(seed)
    X = ggd_sample(rng, N_MC, alpha, beta)
    return fit_exponents(sweep_rows(X))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    n_seeds = N_SEEDS
    if "--seeds" in argv:
        i = argv.index("--seeds")
        n_seeds = max(1, int(argv[i + 1]))

    rng = np.random.default_rng(SEED)
    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    alpha, beta = 12.0, 1.0
    N_MC = 4_000_000
    X = ggd_sample(rng, N_MC, alpha, beta)

    emit("=" * 78)
    emit("Section VII validation: near-parameter detectability chain")
    emit("=" * 78)
    emit("")
    emit("[Step 1+2] separations vs eps = q2/q1 - 1 (coprime pairs)")
    emit("   q1    q2     eps        TV          H^2         chi^2")

    rows = sweep_rows(X)
    for r in rows:
        emit("   %-6d %-6d %-9.4f  %-11.3e %-11.3e %-11.3e"
             % (r[0], r[1], r[2], r[3], r[4], r[5]))

    emit("")
    emit("  fitted exponents (measures ~ eps^p), single seed %d:" % SEED)
    base = fit_exponents(rows)
    for idx, name in ((3, "TV"), (4, "H^2"), (5, "chi^2")):
        predicted = {"TV": 1.0, "H^2": 2.0, "chi^2": 2.0}[name]
        emit("    %-6s exponent = %+.3f   predicted %+.1f   %s"
             % (name, base[name], predicted,
                "OK" if abs(base[name] - predicted) < 0.25 else "CHECK"))

    # ---- replication over independent seeds -------------------------------
    # seed SEED+0 is the run above (same first draw), SEED+1.. are new draws.
    emit("")
    emit("[Step 1+2 replication] %d independent seeds "
         "(one %s-draw GGD sample each)" % (n_seeds, "{:,}".format(N_MC)))
    emit("   seed        TV        H^2      chi^2")
    emit("   %-10d %+8.3f  %+8.3f  %+8.3f"
         % (SEED, base["TV"], base["H^2"], base["chi^2"]))
    reps = [base]
    for s in range(1, n_seeds):
        e = replicate_exponents(SEED + s, alpha, beta, N_MC)
        reps.append(e)
        emit("   %-10d %+8.3f  %+8.3f  %+8.3f"
             % (SEED + s, e["TV"], e["H^2"], e["chi^2"]))
    emit("")
    emit("  mean +/- sd over the %d seeds, with the predicted exponent:" % n_seeds)
    for name, predicted in (("TV", 1.0), ("H^2", 2.0), ("chi^2", 2.0)):
        v = np.array([r[name] for r in reps])
        ok = abs(v.mean() - predicted) < 0.25 and v.std(ddof=1) < 0.15
        emit("    %-6s %+.3f +/- %.3f   predicted %+.1f   sd range "
             "[%+.3f, %+.3f]   %s"
             % (name, v.mean(), v.std(ddof=1), predicted,
                v.min(), v.max(), "OK" if ok else "CHECK"))
    emit("  (seed %d reproduces the original single-seed run exactly, so the" % SEED)
    emit("   values plotted in Fig. 5 are unchanged; only the error bar is new.)")

    emit("")
    emit("[Step 3] Le Cam sample complexity, verified empirically")
    emit("  Actual NP test on finite samples; N_min = smallest N reaching")
    emit("  (miss, FA) <= delta with delta = 0.05, averaged over 20 trials.")
    emit("  NOTE: single seed (%d). This step is not replicated across seeds" % SEED)
    emit("  -- it is ~10x the cost of Step 1+2 and no exponent from it is")
    emit("  quoted in the paper -- so its scatter is not reported.")
    emit("   q1    q2     eps        N_min(measured)   (1-2d)^2/(4H^2)")
    nmin_rows = []
    DELTA = 0.05
    T = 200
    for q1 in (16, 32, 64):
        for target in (0.30, 0.15, 0.08, 0.04):
            q2 = max(q1 + 1, int(math.ceil(q1 * (1 + target))))
            if math.gcd(int(q1), int(q2)) != 1:
                q2 += 1
            eps = q2 / q1 - 1.0
            PD_ref = pmf_on_lattice(quantize(quantize(X, q1), q2), q2)
            P2_ref = pmf_on_lattice(quantize(X, q2), q2)
            _, h2, _ = divergences(PD_ref, P2_ref)
            bound = (1 - 2 * DELTA) ** 2 / (4 * h2)

            eps_f = 1e-14
            log_lr = np.log(PD_ref + eps_f) - np.log(P2_ref + eps_f)

            def lr_stat(vals):
                c = np.floor(vals / q2 + 0.5).astype(np.int64)
                c = np.clip(c, -3000, 3000)
                return float(log_lr[c + 3000].sum())

            reached = None
            for Ntrial in [2 ** k for k in range(8, 19)]:
                s0 = np.empty(T)
                s1 = np.empty(T)
                for t in range(T):
                    s0[t] = lr_stat(quantize(ggd_sample(rng, Ntrial, alpha, beta), q2))
                    s1[t] = lr_stat(quantize(quantize(
                        ggd_sample(rng, Ntrial, alpha, beta), q1), q2))
                thr = float(np.quantile(s0, 1 - DELTA))
                fa = float(np.mean(s0 > thr))
                miss = float(np.mean(s1 <= thr))
                if max(fa, miss) <= DELTA:
                    reached = Ntrial
                    break
            nmin_rows.append((q1, q2, eps, reached, bound))
            emit("   %-6d %-6d %-9.4f  %-17s %-14.3e"
                 % (q1, q2, eps,
                    str(reached) if reached is not None else ">2^18", bound))

    got = [(r[2], r[3]) for r in nmin_rows if r[3]]
    if len(got) >= 3:
        e = np.array([g[0] for g in got]); n = np.array([g[1] for g in got], float)
        p = np.polyfit(np.log(e), np.log(n), 1)
        emit("")
        emit("  fitted N_min ~ eps^q : q = %+.3f   (predicted -2)" % p[0])
        emit("  %s" % ("OK: matches Omega(eps^{-2})" if abs(p[0] + 2) < 0.7
                       else "CHECK: deviation beyond tolerance"))

    emit("=" * 78)
    txt = os.path.join(DATA, "nearparam_validation.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\nwrote " + txt)


if __name__ == "__main__":
    sys.exit(main())
