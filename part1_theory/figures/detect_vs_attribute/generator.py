import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
})

BLUE = '#1f5fb0'
RED = '#c0392b'
GREY = '#444444'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 4.7))

# ---------- Panel (a): Detection, single vs double ----------
x = np.linspace(-4, 6, 400)
p1 = np.exp(-0.5 * ((x - 0.0) / 1.0) ** 2)        # single-compressed
p2 = np.exp(-0.5 * ((x - 2.4) / 1.0) ** 2)        # double-compressed
ax1.plot(x, p1, color=BLUE, lw=2.0, label=r'$P_{\mathrm{single}}$')
ax1.plot(x, p2, color=RED, lw=2.0, label=r'$P_{\mathrm{double}}$')
tau = 1.2
ax1.axvline(tau, color=GREY, ls='--', lw=1.0)
ax1.text(tau + 0.12, 0.86, r'$\tau^\star$', fontsize=8.5, color=GREY)
ax1.fill_between(x, p2, where=(x > tau), color=RED, alpha=0.08)
ax1.fill_between(x, p1, where=(x < tau), color=BLUE, alpha=0.08)
ax1.set_title('(a) Detection: single vs. double recompression', fontsize=9)
ax1.set_xlabel('detection statistic  $S$')
ax1.set_ylabel('density')
ax1.set_yticks([])
ax1.set_xlim(-4, 6)
ax1.legend(loc='upper right', frameon=False, fontsize=8)
ax1.text(0.03, 0.93, r'separable:  $\mathrm{TV}(P_1,P_2)=\Theta(\varepsilon)>0$',
         transform=ax1.transAxes, fontsize=7.3, va='top',
         bbox=dict(boxstyle='round', fc='#eef6ee', ec='#9ec69e', lw=0.6))

# ---------- Panel (b): Attribution, codec A vs B (both double) ----------
xa = np.linspace(-2, 9, 400)
pa = np.exp(-0.5 * ((xa - 3.2) / 1.35) ** 2)       # double via codec A
pb = np.exp(-0.5 * ((xa - 3.38) / 1.35) ** 2)      # double via codec B
ax2.fill_between(xa, np.minimum(pa, pb), color='#7d6b9e', alpha=0.22,
                 lw=0, label=r'overlap $\approx 1$')
ax2.plot(xa, pa, color=BLUE, lw=2.0, label=r'$P_{\mathrm{double}|A}$')
ax2.plot(xa, pb, color=RED, lw=2.0, label=r'$P_{\mathrm{double}|B}$')
ax2.set_title('(b) Attribution: first codec A vs. B (both double)', fontsize=9)
ax2.set_xlabel('statistic  $S$  under  $H_A$  vs.  $H_B$')
ax2.set_ylabel('density')
ax2.set_yticks([])
ax2.set_xlim(-2, 9)
ax2.legend(loc='upper right', frameon=False, fontsize=7.5)
ax2.text(0.03, 0.93, r'indistinguishable:  $\mathrm{TV}\to 0$  as  $\varepsilon\to 0$',
         transform=ax2.transAxes, fontsize=7.3, va='top',
         bbox=dict(boxstyle='round', fc='#fdeeee', ec='#e0a0a0', lw=0.6))
ax2.text(0.03, 0.66, r'$N^\star=\Omega(\varepsilon^{-2})$  samples required',
         transform=ax2.transAxes, fontsize=7.3, va='top',
         bbox=dict(boxstyle='round', fc='#fdeeee', ec='#e0a0a0', lw=0.6))

plt.tight_layout()
plt.savefig('figs/detect_vs_attribute.png', bbox_inches='tight', pad_inches=0.05)
print('saved figs/detect_vs_attribute.png')
