import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8.5,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 6.8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
})

BLUE = '#1f5fb0'
RED = '#c0392b'
GREY = '#777777'

# Reviewer C5 (Fig. 1): every panel must survive black-and-white printing.
# Colour is therefore NEVER the sole channel --- the nested lattices are also
# separated by line style, line width and marker shape, and the two histogram
# populations differ by fill texture (hatch) as well as by tone.

q1 = 4.0
Delta = 2.0
q2 = q1 / Delta

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.45, 2.6))

# ---------- Panel (a): two nested lattices ----------
fine = np.arange(-6, 12 + q2, q2)
coarse = np.arange(-4, 12 + q1, q1)
# fine grid: dotted and thin; coarse grid: solid, thick and drawn last/on top.
ax1.vlines(fine, 0, 1, color=RED, lw=0.7, alpha=0.5, ls=':',
           label='fine grid ($q_2$)')
ax1.vlines(coarse, 0, 1, color=BLUE, lw=2.0, ls='-',
           label='coarse grid ($q_1$)')
# Occupied levels: square markers with a hatch fill --> readable in greyscale.
ax1.scatter(coarse, np.full_like(coarse, 0.5), s=44, marker='s', color='white',
            edgecolor=BLUE, linewidth=1.0, hatch='///', zorder=3)
ax1.set_ylim(-0.05, 1.30)
ax1.set_yticks([])
ax1.set_xlim(-5, 11)
ax1.set_xticks([0, q1, 2 * q1])
ax1.set_xticklabels(['0', '$q_1$', '$2q_1$'])
ax1.set_xlabel('DCT value', fontsize=7.5)
ax1.set_title('(a) nested lattices', fontsize=8)
# Labels sit next to their own element so the two grids cannot be confused.
ax1.text(q1 / 2, 1.10, 'fine step $q_2$', fontsize=6.5, ha='center',
         color=RED)
ax1.text(2 * q1 + 0.15, 1.10, 'coarse step $q_1$', fontsize=6.5, ha='left',
         color=BLUE, va='center')
ax1.legend(loc='lower left', frameon=False, fontsize=6, ncol=1)

# ---------- Panel (b): periodic comb vs uniform single ----------
bins = np.arange(int(round(16 / q2)) + 1)
comb = np.where(bins % Delta == 0, 1.0, 0.12)
xbin = bins * q2 - 6
ax2.axhline(0.5, color=GREY, ls='--', lw=1.0)
# Comb peaks: hatched bars; the low (empty) levels are plain filled bars, so the
# two populations stay distinguishable once colour is removed.
comb_mask = (bins % Delta == 0)
ax2.bar(xbin[comb_mask] - q2 / 2, comb[comb_mask], width=q2 * 0.85,
        facecolor='white', edgecolor=BLUE, lw=0.5, hatch='///',
        label='occupied levels (comb)')
ax2.bar(xbin[~comb_mask] - q2 / 2, comb[~comb_mask], width=q2 * 0.85,
        facecolor=GREY, edgecolor='black', lw=0.3,
        label='empty levels')
ax2.legend(loc='upper right', frameon=False, fontsize=6)
ax2.set_ylim(0, 1.45)
ax2.set_yticks([])
ax2.set_xlim(-5, 11)
ax2.set_xticks([0, q1, 2 * q1])
ax2.set_xticklabels(['0', '$q_1$', '$2q_1$'])
ax2.set_xlabel('quant. level', fontsize=7.5)
ax2.set_title('(b) comb vs. uniform', fontsize=8)

plt.tight_layout(pad=0.3)
plt.savefig('figs/comb_geometry.png', bbox_inches='tight', pad_inches=0.04)
print('saved figs/comb_geometry.png')
