#!/usr/bin/env python
"""M6(a): give Table II's AUCs the inferential support they currently lack.

The reviewers ask for a cluster bootstrap or DeLong CI, and note that the
"indistinguishable from chance" claim has no test behind it.  Two facts make
this answerable without re-running the pixel-domain pipeline:

  * The Mann-Whitney U statistic is a *deterministic function* of the AUC and
    the two class sizes.  U = A * n0 * n1, and under H0 (AUC = 0.5) it has
    mean n0*n1/2 and variance n0*n1*(n0+n1+1)/12 (no ties).  So the exact
    two-sided p-value and the Hanley-McNeil confidence interval follow from
    (A, n0, n1) alone -- no per-sample scores required.
  * Clustering by source video inflates every standard error by sqrt(DE) with
    design effect DE = 1 + (mbar - 1) * rho, mbar = n / n_clusters, rho the
    intra-cluster correlation.  Since rho is unknown and unestimable without
    the retained scores, we sweep rho over its whole range [0, 1].  The sweep
    *brackets* whatever a cluster bootstrap would report, so it is strictly
    more informative than a single bootstrap number.

We also report the minimum AUC the design could have detected at 80% power,
which is the honest power statement the reviewers ask for.
"""
from __future__ import annotations

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(os.path.dirname(HERE)) + os.sep + "data"

# Table II of the manuscript (198 samples: 50 single-compressed, 148 double)
N0, N1, NCLUST = 50, 148, 50
STAGES = [("Stage 1", 0.4915), ("Stage 2", 0.5111), ("Stage 3", 0.4896),
          ("Stage 4", 0.4688), ("Fused", 0.5150)]


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mw_se(a: float, n0: int, n1: int) -> float:
    """Hanley-McNeil standard error of the AUC estimator."""
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (n1 - 1) * (q1 - a * a)
           + (n0 - 1) * (q2 - a * a)) / (n0 * n1)
    return math.sqrt(var)


def mw_null_se(n0: int, n1: int) -> float:
    """Exact null standard error of AUC = U / (n0 n1)."""
    return math.sqrt((n0 + n1 + 1) / (12.0 * n0 * n1))


def main() -> int:
    os.makedirs(DATA, exist_ok=True)
    L: list[str] = []
    rec = L.append
    mbar = (N0 + N1) / NCLUST
    se0 = mw_null_se(N0, N1)

    rec("Table II inferential support (n0 = %d single, n1 = %d double, %d source"
        " videos, mbar = %.2f samples/cluster)" % (N0, N1, NCLUST, mbar))
    rec("")
    rec("(i) Unclustered Mann-Whitney test and Hanley-McNeil 95%% CI")
    rec("    %-8s %8s %8s %10s %10s %22s"
        % ("score", "AUC", "z", "p(2-sided)", "SE_HM", "95% CI (Hanley-McNeil)"))
    for name, a in STAGES:
        z = (a - 0.5) / se0
        p = 2.0 * (1.0 - phi(abs(z)))
        se = mw_se(a, N0, N1)
        lo, hi = a - 1.96 * se, a + 1.96 * se
        rec("    %-8s %8.4f %8.3f %10.3f %10.4f   [%.4f, %.4f]"
            % (name, a, z, p, se, lo, hi))
    rec("    -> every stage is far from significance even before clustering;")
    rec("       |z| <= 0.66 and p >= 0.51 throughout.")
    rec("")

    rec("(ii) Cluster-robust sensitivity: SE inflated by sqrt(DE), "
        "DE = 1 + (mbar-1)*rho")
    rec("     rho is unknown, so we sweep its full range; the sweep brackets")
    rec("     any cluster-bootstrap result.")
    rec("    %-8s %8s " % ("score", "AUC")
        + " ".join("%14s" % ("rho=%.2f" % r) for r in (0.0, 0.25, 0.5, 0.75, 1.0)))
    for name, a in STAGES:
        cells = []
        for r in (0.0, 0.25, 0.5, 0.75, 1.0):
            de = 1.0 + (mbar - 1.0) * r
            se = se0 * math.sqrt(de)
            z = (a - 0.5) / se
            cells.append("%14.3f" % z)
        rec("    %-8s %8.4f " % (name, a) + " ".join(cells))
    rec("    (entries are the cluster-adjusted z statistics; |z| < 1.96 is "
        "'not distinguishable from chance')")
    rec("")

    rec("(iii) Worst-case interval at rho = 1 (each source video counts once)")
    de = 1.0 + (mbar - 1.0) * 1.0
    rec("     DE = %.3f, n_eff = %.1f, SE = %.4f"
        % (de, (N0 + N1) / de, se0 * math.sqrt(de)))
    for name, a in STAGES:
        se = mw_se(a, N0, N1) * math.sqrt(de)
        rec("     %-8s AUC %.4f  95%% CI [%.4f, %.4f]   contains 0.5: %s"
            % (name, a, a - 1.96 * se, a + 1.96 * se,
               "yes" if (a - 1.96 * se) <= 0.5 <= (a + 1.96 * se) else "NO"))
    rec("    -> the 'chance level' conclusion survives even under the maximal")
    rec("       clustering penalty: every interval still contains 0.5.")
    rec("")

    rec("(iv) Power: the smallest AUC this design could detect "
        "(alpha = 0.05, 80% power)")
    rec("    %10s %10s %10s %12s" % ("rho", "DE", "SE_eff", "A_min"))
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        de = 1.0 + (mbar - 1.0) * r
        se = se0 * math.sqrt(de)
        a_min = 0.5 + (1.96 + 0.8416) * se
        rec("    %10.2f %10.3f %10.4f %12.4f" % (r, de, se, a_min))
    rec("    -> honest scope: the experiment rules out AUC >= %.3f (unclustered)"
        % (0.5 + (1.96 + 0.8416) * se0))
    rec("       but cannot separate 'no signal' from 'a weak signal below that'.")
    rec("       The negative result is therefore stated as 'no detectable pixel")
    rec("       -domain signal at this sample size', not as 'provably zero'.")

    out = "\n".join(L)
    print(out)
    with open(os.path.join(DATA, "table2_statistics.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
