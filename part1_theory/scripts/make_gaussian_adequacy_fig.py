"""C2(ii): a minimal, self-contained adequacy check of the local Gaussian model.

Part I assumes per-stage local Gaussian laws:
    Stage 1   q | H_i  ~ N
    Stage 2   h | H_1  ~ N(mu_0 + a, Sigma)
    Stage 3   D | H_i  ~ N
Reviewer C2 complains that Part I gives no evidence for these at all ("万年?").
This script produces ONE figure, reproducible from the archived CSV already in
the repository, comparing each stage's empirical score distribution under H_0
(single compression) with the Gaussian fitted to that same sample.

It deliberately does NOT claim the model is correct.  It reports two honest
descriptive statistics (skewness and excess kurtosis with a bootstrap 95% CI),
so the figure can be captioned with numbers rather than adjectives.

Run:  python scripts/theory/make_gaussian_adequacy_fig.py
Data: data/pixel_domain_rerun.csv   (relative to the repository root)
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(ROOT, "figs")
DATA = os.path.join(ROOT, "data", "pixel_domain_rerun.csv")
SEED = 20261022

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "legend.fontsize": 6.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

STAGES = ["S1", "S3"]           # Stage scores available per video
PT = 10_000                     # draws for the fitted reference
B = 4_000                       # bootstrap replicates
CI = 95


def load():
    """Return {stage: {label: array}} from the archived per-sample CSV."""
    import csv
    out = {s: {} for s in STAGES}
    rows = []
    with open(DATA, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    labs = sorted({int(float(r["label"])) for r in rows})
    for s in STAGES:
        for L in labs:
            vals = [float(r[s]) for r in rows
                    if int(float(r["label"])) == L
                    and r[s] not in ("", "nan")]
            out[s][L] = np.asarray(vals, dtype=float)
    return out


def moments(x):
    """Skewness and excess kurtosis (Fisher), NaN-safe."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return np.nan, np.nan
    m = x.mean()
    sd = x.std(ddof=1)
    if sd == 0:
        return np.nan, np.nan
    z = (x - m) / sd
    g1 = np.mean(z ** 3)
    g2 = np.mean(z ** 4) - 3.0
    return float(g1), float(g2)


def boot_ci(x, fn, rng, b=B, ci=CI):
    """Percentile bootstrap CI for a scalar statistic."""
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return (np.nan, np.nan)
    idx = rng.integers(0, x.size, size=(b, x.size))
    samp = x[idx]
    m = samp.mean(axis=1, keepdims=True)
    sd = samp.std(axis=1, ddof=1, keepdims=True)
    sd = np.where(sd == 0, np.nan, sd)
    z = (samp - m) / sd
    stat = fn(z)
    if np.all(np.isnan(stat)):
        return (np.nan, np.nan)
    lo = (100 - ci) / 2.0
    return tuple(np.nanpercentile(stat, [lo, 100 - lo]))


def main():
    if not os.path.exists(DATA):
        print("ERROR: missing data file %s" % DATA)
        return 1
    rng = np.random.default_rng(SEED)
    data = load()

    ncol = len(STAGES)
    fig, axes = plt.subplots(1, ncol, figsize=(3.45, 1.65))
    if ncol == 1:
        axes = [axes]

    report = []
    for ax, s in zip(axes, STAGES):
        for L, style in ((0, dict(color="#1f5fb0", ls="-")),
                         (1, dict(color="#c0392b", ls="-"))):
            x = data[s].get(L, np.array([]))
            if x.size < 4:
                continue
            mu, sd = x.mean(), x.std(ddof=1)
            # reference Gaussian, drawn on the same n for visual honesty
            ref = rng.normal(mu, sd, PT)
            lo, hi = np.percentile(np.concatenate([x, ref]), [0.5, 99.5])
            grid = np.linspace(lo, hi, 200)
            ax.plot(grid, gaussian_pdf(grid, mu, sd), ls=style["ls"],
                    lw=1.0, color=style["color"], alpha=0.85,
                    label=("Gaussian fit, $H_%d$" % L))
            ax.hist(x, bins=8, density=True, histtype="step", lw=1.2,
                    color=style["color"], alpha=0.9,
                    label=("empirical $H_%d$ ($n=%d$)" % (L, x.size)))
            if L == 0:
                g1, g2 = moments(x)
                c1 = boot_ci(x, lambda z: np.mean(z ** 3, axis=1), rng)
                c2 = boot_ci(x, lambda z: np.mean(z ** 4, axis=1) - 3.0, rng)
                report.append((s, x.size, g1, c1, g2, c2))
        ax.set_title("%s: score vs. fitted Gaussian" % s, fontsize=7)
        ax.set_xlabel("score", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.legend(fontsize=5.6, frameon=False)
        ax.tick_params(labelsize=6)

    fig.tight_layout(pad=0.6)
    outpng = os.path.join(FIGS, "gaussian_adequacy.png")
    fig.savefig(outpng, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved %s" % outpng)
    print("\n%-4s %5s %8s %-22s %8s %-22s" %
          ("col", "n", "skew", "skew 95% CI", "exkurt", "exkurt 95% CI"))
    for s, n, g1, c1, g2, c2 in report:
        print("%-4s %5d %8.3f [%6.3f, %6.3f]      %8.3f [%6.3f, %6.3f]"
              % (s, n, g1, c1[0], c1[1], g2, c2[0], c2[1]))
    return 0


def gaussian_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


if __name__ == "__main__":
    sys.exit(main())
