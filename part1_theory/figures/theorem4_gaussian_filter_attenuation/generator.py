"""Regenerate the simulation-based figures of the companion theory paper.

Every panel is produced from the exact statement it illustrates, with a fixed seed,
so the figure files are reproducible.  One figure (near_parameter_real.png) needs the
video corpus and is kept as a legacy artefact; see figs/README.md.

Metric convention.  The periodicity of a comb with period Delta is measured by the
*fundamental-to-DC ratio*

    R_Delta(h) = |hat h(L / Delta)| / hat h(0),

not by a peak-to-mean score.  The older peak-to-mean score is confounded by comb
sparsity (Remark rem:two_regimes: sparsity is geometry, not detectability): doubling
Delta halves the number of occupied bins, which halves the spectral mean and therefore
*lowers* the peak-to-mean score even though the comb is stronger.  R_Delta is exactly
the quantity that Proposition prop:spectral_comb predicts (|hat h(M)| = hat h(0) = N)
and that Theorem thm:lowpass_attenuation predicts to decay as exp(-2 pi^2 sigma^2 / Delta^2).

Run:  python scripts/theory/make_figures.py
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)
SEED = 20261005

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     # Times New Roman for ALL in-figure text (matches the
                     # IEEE body font); STIX for math (Times-like).
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif",
                                    "Liberation Serif", "serif"],
                     "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "ps.fonttype": 42})


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def quantize(x, q):
    return q * np.floor(x / q + 0.5)


def comb_ratio(h, delta):
    """R_Delta(h) = |hat h(L/Delta)| / hat h(0);  L must be a multiple of Delta."""
    L = h.size
    assert L % delta == 0, (L, delta)
    hat = np.fft.fft(h)
    return float(np.abs(hat[L // delta]) / np.abs(hat[0]))


def gaussian1d(sigma, radius=None):
    if sigma <= 0:
        return np.array([1.0])
    r = int(radius or max(4, math.ceil(4 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gauss_transfer(f_hat, sigma, jmax=4):
    """DFT of the *sampled* Gaussian kernel at normalized frequency f_hat.

    Sampling in space periodizes the spectrum, so the discrete-kernel transfer
    function is the aliasing sum  sum_j G(f_hat + j) / sum_j G(j)  with
    G(u) = exp(-2 pi^2 sigma^2 u^2).  The continuous-Gaussian prediction
    exp(-2 pi^2 sigma^2 f_hat^2) is the j = 0 term alone; the two differ most at
    high frequency (notably at Nyquist, f_hat = 1/2).
    """
    if sigma <= 0:
        return 1.0
    c = -2.0 * math.pi ** 2 * sigma ** 2
    num = sum(math.exp(c * (f_hat + j) ** 2) for j in range(-jmax, jmax + 1))
    den = sum(math.exp(c * j ** 2) for j in range(-jmax, jmax + 1))
    return num / den


def ggd_sample(rng, n, alpha, beta):
    g = rng.gamma(1.0 / beta, 1.0, size=n)
    return alpha * np.power(g, 1.0 / beta) * rng.choice([-1.0, 1.0], size=n)


def bins_of(values, width, L):
    idx = np.floor(values / width).astype(np.int64) + L // 2
    idx = idx[(idx >= 0) & (idx < L)]
    return np.bincount(idx, minlength=L).astype(float)


def pmf_on_lattice(values, q, width=6001):
    c = np.floor(values / q + 0.5).astype(np.int64)
    c = np.clip(c, -(width // 2), width // 2)
    z = np.bincount(c + width // 2, minlength=width).astype(float)
    return z / z.sum()


def divergences(p, q):
    tv = 0.5 * float(np.abs(p - q).sum())
    h2 = float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    m = (p + q) / 2.0
    nz = m > 1e-15
    chi2 = float(np.sum((p[nz] - q[nz]) ** 2 / m[nz]))
    return tv, h2, chi2


# --------------------------------------------------------------------------- #
# log-axis tick labels
# --------------------------------------------------------------------------- #
# On a *short* log range (here the epsilon axis spans barely one decade) matplotlib
# falls back to labelling the minor ticks as well as the major ones.  On a half-page
# panel that prints every 10^-2 multiple from 3e-2 to 9e-2 followed by 2e-1..4e-1,
# and neighbouring labels then collide ("3 x 10^-1" printed on top of "4 x 10^-1").
# We therefore pin an explicit, sparse major-tick set and silence the minor labels;
# the minor ticks themselves stay so the log gridlines survive.
def _log_label(v):
    """Compact ``a x 10^b`` mathtext label for one log-axis tick value.

    The mantissa is normalised into ``[1, 10)`` -- ``0.04`` has to read
    ``4 x 10^-2``, not ``0.4 x 10^-1``.
    """
    e = math.floor(math.log10(v) + 1e-12)
    m = v / (10.0 ** e)
    if abs(m - 1.0) < 1e-9:
        return "$10^{%d}$" % e
    return "$%g\\!\\times\\!10^{%d}$" % (m, e)


def set_log_x_ticks(ax, ticks):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter([_log_label(v) for v in ticks]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def xtick_labels_overlap(ax, renderer=None):
    """Report overlapping x tick labels of ``ax`` (for the self-check below).

    Returns the list of colliding label pairs, measured from the layout boxes.
    """
    fig = ax.figure
    if renderer is None:
        renderer = fig.canvas.get_renderer()
    boxes = []
    for lab in ax.get_xticklabels():
        if not lab.get_text():
            continue
        try:
            bb = lab.get_window_extent(renderer=renderer)
        except Exception:
            continue
        boxes.append((lab.get_text(), bb))
    boxes.sort(key=lambda t: t[1].x0)
    bad = []
    for (t0, b0), (t1, b1) in zip(boxes, boxes[1:]):
        if b1.x0 < b0.x1 - 0.5:            # half a pixel of slack
            bad.append((t0, t1, round(b0.x1 - b1.x0, 1)))
    return bad


# --------------------------------------------------------------------------- #
# Fig. 1  Theorem 1: comb structure, q1 = 2, q2 = 4 -> g = 2, Delta = 2
# --------------------------------------------------------------------------- #
def fig_periodicity():
    rng = np.random.default_rng(SEED)
    q1, q2 = 2, 4
    g = math.gcd(q1, q2)
    b = q2 // g
    L = 256
    X = 12.0 * rng.exponential(1.0, 400_000) * rng.choice([-1.0, 1.0], 400_000)
    h1 = bins_of(quantize(X, q1), g, L)
    h2 = bins_of(quantize(quantize(X, q1), q2), g, L)
    r1, r2 = comb_ratio(h1, b), comb_ratio(h2, b)

    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    lo, hi = 108, 148
    ax[0][0].bar(np.arange(lo, hi), h1[lo:hi], width=0.9, alpha=.6, label="single")
    ax[0][0].bar(np.arange(lo, hi), h2[lo:hi], width=0.5, color="crimson", label="double")
    ax[0][0].set_title("Histograms (bins of width $g=%d$)" % g)
    ax[0][0].legend()

    ax[0][1].bar(np.arange(lo, hi), h2[lo:hi], width=0.9, color="crimson")
    ax[0][1].set_title("Bin-index histogram of the double-quantized values")
    ax[0][1].set_xlabel("bin index $k$ (width $g$)")

    z = h2 - h2.mean()
    ac = np.correlate(z, z, "full")[len(z) - 1:]
    ac /= ac[0]
    ax[1][0].plot(np.arange(0, 25), ac[:25], "o-", ms=3)
    for p in range(b, 25, b):
        ax[1][0].axvline(p, color="k", ls=":", lw=.8)
    ax[1][0].set_title("Autocorrelation (dotted = multiples of $\\Delta=%d$)" % b)
    ax[1][0].set_xlabel("lag (bins)")

    f = np.fft.rfftfreq(L)
    m1 = np.abs(np.fft.rfft(h1)) / np.abs(np.fft.fft(h1)[0])
    m2 = np.abs(np.fft.rfft(h2)) / np.abs(np.fft.fft(h2)[0])
    ax[1][1].plot(f, m1, lw=1, label="single  ($R_\\Delta=%.4f$)" % r1)
    ax[1][1].plot(f, m2, lw=1, color="crimson", label="double  ($R_\\Delta=%.4f$)" % r2)
    ax[1][1].axvline(1.0 / b, color="k", ls="--", lw=.8,
                     label="$1/\\Delta=%.2f$ cycles/bin" % (1.0 / b))
    ax[1][1].set_yscale("log")
    ax[1][1].set_title("FFT magnitude normalized by DC")
    ax[1][1].set_xlabel("cycles / bin")
    ax[1][1].legend(fontsize=7)
    fig.suptitle("Theorem 1 on synthetic DCT coefficients: $q_1=%d$, $q_2=%d$ "
                 "($g=%d$, $\\Delta=%d$)" % (q1, q2, g, b))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "periodicity_q2_q4.png"))
    plt.close(fig)
    return "periodicity_q2_q4.png  R_single=%.4f  R_double=%.4f" % (r1, r2)


# --------------------------------------------------------------------------- #
# Fig. 2  Theorem 2 / Prop: max-pooling vs averaging, R = 64
# --------------------------------------------------------------------------- #
def fig_max_pooling():
    rng = np.random.default_rng(SEED + 1)
    R, DELTA, TRIALS = 64, 3.0, 4000
    ms = [1, 2, 4, 8, 16, 32, 48, 64]

    # calibrate both statistics on the null so that the false-alarm rate is 5 %
    null = rng.normal(0, 1, (20_000, R))
    tau_max = float(np.quantile(null.max(1), 0.95))
    tau_avg = float(np.quantile(null.mean(1), 0.95))

    dmax, davg = [], []
    for m in ms:
        s0 = rng.normal(0, 1, (TRIALS, R))
        s1 = s0.copy()
        s1[:, :m] = rng.normal(DELTA, 1, (TRIALS, m))
        dmax.append(float(np.mean(s1.max(1) > tau_max)) - float(np.mean(s0.max(1) > tau_max)))
        davg.append(float(np.mean(s1.mean(1) > tau_avg)) - float(np.mean(s0.mean(1) > tau_avg)))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ms, dmax, "o-", label="max-pooling  $S_{\\max}$")
    ax.plot(ms, davg, "s--", label="global averaging  $S_{\\mathrm{avg}}$")
    # Reviewer B8: the axis is (power - size); the horizontal line at 0 IS the
    # size baseline (chance line). Draw it explicitly and label it so it is not
    # mistaken for the x-axis spine.
    ax.axhline(0, color="gray", ls="--", lw=1.1, label="size baseline (power = size)")
    ax.annotate("chance / size line", xy=(ms[-1], 0.0), xytext=(ms[-1], 0.045),
                fontsize=7, color="gray", ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
    ax.set_xlabel("number of tampered regions $m$ (of $R=64$)")
    ax.set_ylabel("detection advantage (power $-$ size, size $=0.05$)")
    ax.set_title("Theorem 2 and Prop.: max-pooling retains a positive advantage\n"
                 "where averaging is diluted by the factor $m/R$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "max_pooling_vary_m.png"))
    plt.close(fig)
    return ("max_pooling_vary_m.png  tau_max=%.3f tau_avg=%.3f  "
            "Dmax(m=1)=%.3f  Davg(m=1)=%.3f  Davg(m=64)=%.3f"
            % (tau_max, tau_avg, dmax[0], davg[0], davg[-1]))


# --------------------------------------------------------------------------- #
# Fig. 3  Theorem 4: Gaussian low-pass attenuation
# --------------------------------------------------------------------------- #
def fig_theorem4():
    rng = np.random.default_rng(SEED + 2)
    q1, q2, g = 4, 8, 4
    b = q2 // g
    L, trials = 256, 20
    sigmas = np.linspace(0.0, 2.0, 21)
    ratios = []
    for s in sigmas:
        acc = []
        for _ in range(trials):
            X = 12.0 * rng.exponential(1.0, 200_000) * rng.choice([-1.0, 1.0], 200_000)
            h = bins_of(quantize(quantize(X, q1), q2), g, L)
            if s > 0:
                h = np.convolve(h, gaussian1d(s), mode="same")
            acc.append(comb_ratio(h, b))
        ratios.append(np.mean(acc))
    ratios = np.array(ratios)
    base = ratios[0]
    ratios = ratios / base
    theory = np.exp(-2 * np.pi ** 2 * sigmas ** 2 / b ** 2)      # continuous Gaussian
    alias = np.array([gauss_transfer(1.0 / b, s) for s in sigmas])  # sampled kernel

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(sigmas, ratios, "o-", ms=3, label="measured $R_\\Delta(\\tilde h)/R_\\Delta(h)$")
    ax.semilogy(sigmas, theory, "--", color="green",
                label="$\\exp(-2\\pi^2\\sigma^2/\\Delta^2)$, $\\Delta=%d$" % b)
    ax.semilogy(sigmas, alias, ":", color="black",
                label="same, with kernel aliasing")
    ax.set_xlabel("Gaussian filter strength $\\sigma$ (bins)")
    ax.set_ylabel("fundamental-to-DC ratio, normalized")
    ax.set_title("Theorem 4: attenuation of the periodic signature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "theorem4_gaussian_filter_attenuation.png"))
    plt.close(fig)
    err = float(np.max(np.abs(ratios - theory)))
    err2 = float(np.max(np.abs(ratios - alias)))
    return ("theorem4_gaussian_filter_attenuation.png  R(0)=%.4f  at s=1.0: "
            "measured %.5f / continuous %.5f / aliased %.5f ; max|dev| %.4f vs %.4f"
            % (base, ratios[10], theory[10], alias[10], err, err2))


# --------------------------------------------------------------------------- #
# Fig. 4  Theorem 5 / Prop: fused AUC vs number of stages K
# --------------------------------------------------------------------------- #
def fig_theorem5():
    rng = np.random.default_rng(SEED + 3)
    Ks = list(range(1, 11))
    per_stage_auc, fused_auc, fused_sd = [], [], []
    n0 = n1 = 4000
    for K in Ks:
        a_s, a_f = [], []
        for _ in range(20):
            x0 = rng.normal(0, 1, (n0, K))
            x1 = rng.normal(0.9, 1, (n1, K))
            aucs = [float(np.mean(x1[:, k][:, None] > x0[:, k][None, :]) +
                          0.5 * np.mean(x1[:, k][:, None] == x0[:, k][None, :]))
                    for k in range(K)]
            a_s.append(np.mean(aucs))
            s0, s1 = x0.mean(1), x1.mean(1)
            a_f.append(np.mean(s1[:, None] > s0[None, :]) +
                       0.5 * np.mean(s1[:, None] == s0[None, :]))
        per_stage_auc.append(np.mean(a_s))
        fused_auc.append(np.mean(a_f))
        fused_sd.append(np.std(a_f))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(Ks, fused_auc, yerr=fused_sd, fmt="o-", ms=4, capsize=2,
                label="fused")
    ax.plot(Ks, per_stage_auc, "s--", label="average single stage")
    ax.set_xlabel("number of conditionally independent stages $K$")
    ax.set_ylabel("AUC")
    ax.set_title("Theorem 5 and Prop.: fusion improves monotonically with $K$")
    ax.legend(loc="lower right")
    # Reviewer C5 (Fig. 5): the per-stage AUC is deliberately ~constant in $K$
    # (each stage is conditionally independent and of equal quality), so the
    # single-stage curve looks flat on the full [0.7,1] range and can be misread
    # as a single point. Add a magnified inset so the curve is seen to be a line.
    axins = ax.inset_axes([0.07, 0.52, 0.42, 0.40])
    axins.errorbar(Ks, fused_auc, yerr=fused_sd, fmt="o-", ms=3, capsize=1.5,
                   label="fused")
    axins.plot(Ks, per_stage_auc, "s--", label="avg single stage")
    axins.set_xlim(0.5, 10.5)
    axins.set_ylim(0.70, 0.76)
    axins.set_xlabel("$K$", fontsize=7)
    axins.set_ylabel("AUC", fontsize=7)
    axins.tick_params(labelsize=6)
    axins.set_title("single-stage detail", fontsize=7)
    axins.legend(fontsize=6, loc="lower right", frameon=False)
    ax.indicate_inset_zoom(axins, edgecolor="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "theorem5_vary_K_auc.png"))
    plt.close(fig)
    return ("theorem5_vary_K_auc.png  per-stage=%.3f  fused(K=1)=%.3f  fused(K=10)=%.3f"
            % (per_stage_auc[0], fused_auc[0], fused_auc[-1]))


# --------------------------------------------------------------------------- #
# Fig. 5  Section VII: near-parameter behaviour on synthetic GGD coefficients
# --------------------------------------------------------------------------- #
def fig_nearparam_synthetic():
    """Section VII: the barrier is DISTRIBUTIONAL, not spectral.

    Panels show the three links of the chain
        d_TV = Theta(eps)   ->   H^2, chi^2 = O(eps^2)   ->   N* = Omega(eps^{-2}),
    all measured on one common GGD sample, plus the two pmfs being separated.
    """
    rng = np.random.default_rng(SEED + 4)
    alpha, beta, N_MC = 12.0, 1.0, 4_000_000
    X = ggd_sample(rng, N_MC, alpha, beta)

    rows, pmfs = [], []
    for q1 in (16, 32, 64):
        for target in (0.40, 0.25, 0.15, 0.08, 0.04, 0.02):
            q2 = max(q1 + 1, int(math.ceil(q1 * (1 + target))))
            if math.gcd(q1, q2) != 1:
                q2 += 1
            e = q2 / q1 - 1.0
            if e <= 0 or e > 0.5:
                continue
            PD = pmf_on_lattice(quantize(quantize(X, q1), q2), q2)
            P2 = pmf_on_lattice(quantize(X, q2), q2)
            tv, h2, c2 = divergences(PD, P2)
            rows.append((q1, q2, e, tv, h2, c2))
            pmfs.append((q1, q2, e, PD, P2))
    rows.sort(key=lambda r: r[2])
    eps = np.array([r[2] for r in rows])
    tvs = np.array([r[3] for r in rows])
    h2s = np.array([r[4] for r in rows])
    chi2s = np.array([r[5] for r in rows])

    def fitp(x, y):
        return float(np.polyfit(np.log(x), np.log(y), 1)[0])

    def ref(x, p, anchor):
        return anchor * (x / x[0]) ** p

    p_tv, p_h2, p_c2 = fitp(eps, tvs), fitp(eps, h2s), fitp(eps, chi2s)
    DELTA = 0.05
    n_bound = (1 - 2 * DELTA) ** 2 / (4 * h2s)          # Le Cam, from measured H^2

    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    ax[0][0].loglog(eps, tvs, "o", ms=4, label="measured $d_{\\mathrm{TV}}$")
    ax[0][0].loglog(eps, ref(eps, 1.0, tvs[0]), "--", color="gray",
                    label="$\\propto\\varepsilon$")
    ax[0][0].set_xlabel("$\\varepsilon = q_2/q_1 - 1$")
    ax[0][0].set_ylabel("$d_{\\mathrm{TV}}(P_D,P_2)$")
    ax[0][0].set_title("total variation (fitted exponent %+.3f, theory $+1$)" % p_tv)
    ax[0][0].legend()

    ax[0][1].loglog(eps, h2s, "o", ms=4, label="$H^2$")
    ax[0][1].loglog(eps, chi2s, "s", ms=4, label="$\\chi^2$")
    ax[0][1].loglog(eps, ref(eps, 2.0, h2s[0]), "--", color="gray",
                    label="$\\propto\\varepsilon^2$")
    ax[0][1].set_xlabel("$\\varepsilon = q_2/q_1 - 1$")
    ax[0][1].set_title("Hellinger / $\\chi^2$ (fitted %+.3f / %+.3f, theory $+2$)"
                       % (p_h2, p_c2))
    ax[0][1].legend()

    ax[1][0].loglog(eps, n_bound, "o-", ms=4, color="crimson",
                    label="Le Cam bound $(1-2\\delta)^2/(4H^2)$")
    ax[1][0].loglog(eps, ref(eps, -2.0, n_bound[0]), "--", color="gray",
                    label="$\\propto\\varepsilon^{-2}$")
    ax[1][0].set_xlabel("$\\varepsilon = q_2/q_1 - 1$")
    ax[1][0].set_ylabel("required sample size $N$")
    ax[1][0].set_title("Theorem 6: sample complexity ($\\delta=0.05$, from measured $H^2$)")
    ax[1][0].legend()

    for (q1, q2, e, PD, P2), col in ((pmfs[0], "tab:blue"), (pmfs[-1], "tab:orange")):
        c0 = np.flatnonzero((PD > 1e-8) | (P2 > 1e-8))
        lo, hi = max(0, c0.min() - 2), min(PD.size, c0.max() + 3)
        kk = np.arange(lo, hi) - PD.size // 2
        ax[1][1].plot(kk, PD[lo:hi], "-", color=col, lw=1,
                      label="$P_D$, $q_1=%d,q_2=%d$" % (q1, q2))
        ax[1][1].plot(kk, P2[lo:hi], ":", color=col, lw=1.4,
                      label="$P_2$, $q_1=%d,q_2=%d$" % (q1, q2))
    ax[1][1].set_yscale("log")
    ax[1][1].set_xlabel("output cell index $n$ (units of $q_2$)")
    ax[1][1].set_ylabel("probability")
    ax[1][1].set_title("double vs single quantization pmf")
    ax[1][1].legend(fontsize=7)

    # x axis of the three log-x panels: epsilon only spans [0.03, 0.44], i.e. just
    # over one decade, and on such a short log range matplotlib also labels the
    # minor ticks -- 3e-2 and 4e-2 (and 3e-1 and 4e-1) then print on top of each
    # other.  Pin a sparse major set and keep the minor ticks unlabelled.
    ticks = [0.04, 0.06, 0.1, 0.2, 0.4]
    for a in (ax[0][0], ax[0][1], ax[1][0]):
        set_log_x_ticks(a, ticks)

    fig.suptitle("Near-parameter behaviour on synthetic GGD coefficients "
                 "($\\alpha=%.0f$, $\\beta=%.0f$, $n=4\\times10^6$)" % (alpha, beta))
    fig.tight_layout()
    fig.canvas.draw()
    clashes = []
    for a in (ax[0][0], ax[0][1], ax[1][0]):
        clashes += xtick_labels_overlap(a)
    if clashes:
        print("    !! overlapping x tick labels: %s" % clashes, file=sys.stderr)
    fig.savefig(os.path.join(FIGS, "near_parameter_synthetic.png"))
    plt.close(fig)
    return ("near_parameter_synthetic.png  %d points, eps in [%.3f, %.3f]; "
            "exponents TV %+.3f / H2 %+.3f / chi2 %+.3f ; N* %.3g -> %.3g"
            % (len(eps), eps.min(), eps.max(), p_tv, p_h2, p_c2,
               n_bound.min(), n_bound.max()))


# --------------------------------------------------------------------------- #
# Fig. 6  Lemma / Theorem 7: null tail and the platform region
# --------------------------------------------------------------------------- #
def fig_theorem7():
    rng = np.random.default_rng(SEED + 5)
    Ts = np.linspace(0.0, 0.8, 60)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s0 in (0.05, 0.10, 0.20, 0.30):
        x = rng.normal(0.0, s0, 200_000)
        ax.semilogy(Ts, [max(float(np.mean(np.abs(x) > t)), 1e-5) for t in Ts],
                    lw=1.2, label="$\\sigma_0=%.2f$" % s0)
    ax.axvline(0.195, color="k", ls="--", lw=1)
    ax.text(0.20, 1e-3, " $T=0.195$", fontsize=8)
    ax.set_xlabel("threshold $T$")
    ax.set_ylabel("$P_0(S>T)$")
    ax.set_title("Lemma and Theorem 7: null tail and the platform region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "theorem7_platform_region.png"))
    plt.close(fig)
    return "theorem7_platform_region.png"


# --------------------------------------------------------------------------- #
# Fig. 7  Theorem 8: proxy error vs deblocking strength
# --------------------------------------------------------------------------- #
def fig_theorem8():
    rng = np.random.default_rng(SEED + 6)
    q1, q2, g = 4, 8, 4
    b = q2 // g
    L = 256
    X = 12.0 * rng.exponential(1.0, 300_000) * rng.choice([-1.0, 1.0], 300_000)
    h = bins_of(quantize(quantize(X, q1), q2), g, L)
    base = comb_ratio(h, b)
    strengths = np.linspace(0.0, 2.0, 21)
    dev, retained = [], []
    for s in strengths:
        hs = np.convolve(h, gaussian1d(s), mode="same") if s > 0 else h.copy()
        r = comb_ratio(hs, b)
        dev.append(1.0 - r / base)          # fraction of signature erased
        retained.append(np.exp(-2 * np.pi ** 2 * s ** 2 / b ** 2))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(strengths, dev, "o-", ms=3, color="red", label="measured erasure")
    ax.plot(strengths, retained, "--", color="green", label="theory (retained)")
    ax.set_xlabel("deblocking strength (Gaussian $\\sigma$, bins)")
    ax.set_ylabel("erased / retained fraction of the comb fundamental")
    ax.set_title("Theorem 8: proxy error grows as the signature is erased")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "theorem8_deblock_strength.png"))
    plt.close(fig)
    return ("theorem8_deblock_strength.png  R(0)=%.4f  erased@1.0=%.4f  "
            "retained@1.0=%.4f" % (base, dev[10], retained[10]))


def main():
    figs = (fig_periodicity, fig_max_pooling, fig_theorem4, fig_theorem5,
            fig_nearparam_synthetic, fig_theorem7, fig_theorem8)
    # Optional filter so one panel can be re-rendered while iterating:
    #   python make_figures.py nearparam        # only Fig. 5
    # With no argument every figure is rebuilt (the default, and what the
    # manuscript build does).
    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    for fn in figs:
        if sel and not any(s.lower() in fn.__name__.lower() for s in sel):
            continue
        print("  " + fn(), flush=True)
    print("\nfigures written to", FIGS)


if __name__ == "__main__":
    sys.exit(main())
