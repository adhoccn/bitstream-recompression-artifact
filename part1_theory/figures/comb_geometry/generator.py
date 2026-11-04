import os

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
TITLE_PAD = 10.0   # clearance between a panel title and its frame
VLINE_TOP = 1.0   # every vline stops here; grid names live just above it
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
ax1.vlines(fine, 0, 1, color=RED, lw=0.7, alpha=0.5, ls=':')
ax1.vlines(coarse, 0, 1, color=BLUE, lw=2.0, ls='-')
# Occupied levels: square markers with a hatch fill --> readable in greyscale.
ax1.scatter(coarse, np.full_like(coarse, 0.5), s=44, marker='s', color='white',
            edgecolor=BLUE, linewidth=1.0, hatch='///', zorder=3)
# -------- grid names: written next to the lines they name --------
# A legend in the corner cannot disambiguate these two grids: both lattices
# fill the entire panel, so a colour swatch has nothing local to attach to and
# the reader must match it against every line in the figure. Attach each name
# to one specific line with a leader instead; colour then confirms rather than
# carries the meaning, so the panel also survives greyscale printing (the two
# grids already differ by style: dotted-thin vs solid-thick).
# Vertical bands, bottom-up: vlines end at 1.00; grid NAMES occupy ~1.03-1.10;
# the fine step arrow sits at 1.24; the coarse step arrow at 1.46. Keeping the
# bands separate is what stops a label from landing on a line or on another
# label -- all three defects reported for this panel were band collisions.
def grid_label(ax, x_line, text, color, dy, xoff=0.0, ha='center'):
    """Name one grid by pointing at a single line that belongs to it."""
    # Sit just above the vline tops (y=1.0) but below the step arrows, and pick
    # x values well clear of [0, q1] where the arrows live, so nothing collides.
    y = VLINE_TOP + dy
    ax.annotate(text, xy=(x_line, VLINE_TOP), xytext=(x_line + xoff, y),
                fontsize=6.2, ha=ha, va='bottom', color=color,
                arrowprops=dict(arrowstyle='-', color=color, lw=0.6,
                                alpha=0.9, shrinkA=0, shrinkB=1))


ax1.set_ylim(-0.05, 1.62)
ax1.set_yticks([])
ax1.set_xlim(-5, 11)
ax1.set_xticks([0, q1, 2 * q1])
ax1.set_xticklabels(['0', '$q_1$', '$2q_1$'])
ax1.set_xlabel('DCT value', fontsize=7.5)
ax1.set_title('(a) nested lattices', fontsize=8, pad=TITLE_PAD)


# Dimension lines. Each label is carried by a two-headed arrow spanning exactly
# the interval it names, so there is no ambiguity about which grid it refers to.
# Placing free text at y=1.10 previously sat it directly on top of a fine-grid
# line (the fine lattice has step q2=2, so every even x carries a line, and
# x=q1/2=2 landed squarely on one); the coarse label was also anchored at
# x=2*q1+0.15, far from the span it describes, and ran past the right edge.
def dim(ax, x0, x1, y, text, color):
    """Two-headed dimension arrow spanning [x0,x1], labelled above its middle."""
    ax.annotate('', xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='<->', color=color,
                                lw=0.9, shrinkA=0, shrinkB=0))
    ax.plot([x0, x0], [y - 0.045, y + 0.045], color=color, lw=0.7)
    ax.plot([x1, x1], [y - 0.045, y + 0.045], color=color, lw=0.7)
    ax.text((x0 + x1) / 2, y + 0.075, text, fontsize=6.5, ha='center',
            va='bottom', color=color)


# -------- v1 layout restored per the author's request --------
# The statements below are the ones the author approved; keep them exactly.
# fine step q2 occupies [0, q2]; coarse step q1 occupies [0, q1]. Two different
# heights keep the arrows from overlapping each other.
dim(ax1, 0, q2, 1.24, 'fine step $q_2$', RED)
dim(ax1, 0, q1, 1.46, 'coarse step $q_1$', BLUE)

# Point at one line per grid, chosen where that grid is visually dominant and
# away from the step arrows so nothing collides.
# Anchor each name on a line that belongs to that grid ONLY. The coarse lines
# sit at multiples of q1 (-4, 0, 4, 8, ...); the fine ones at multiples of q2,
# i.e. every even value. So x=-4 is coarse-only, while x=10 is fine-only (8 is
# disqualified because it is also coarse). Anything else would be ambiguous.
# Growth direction is chosen per label so each stays inside the frame: the
# coarse name starts at x=-4 and grows rightwards, the fine name ends at x=10
# and grows leftwards. Both are ~3.4 data units wide against a [-5,11] axis.
grid_label(ax1, -1.0 * q1, 'coarse grid ($q_1$)', BLUE, dy=0.07, xoff=0.05,
           ha='left')
grid_label(ax1, 10.0, 'fine grid ($q_2$)', RED, dy=0.07, xoff=-0.05,
           ha='right')
# No corner legend for panel (a): the names are now on the lines themselves.

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
ax2.set_title('(b) comb vs. uniform', fontsize=8, pad=TITLE_PAD)

plt.tight_layout(pad=0.3)
# Push the drawing area down a little. With the default top the panels sat
# almost flush against their titles (about 18 px of clearance on a 780 px
# canvas) while the lower part of the frame carried unused space; moving the
# axes region down restores breathing room above each panel and re-centres the
# content vertically.
fig.subplots_adjust(top=0.90)
os.makedirs('figs', exist_ok=True)
plt.savefig('figs/comb_geometry.png', bbox_inches='tight', pad_inches=0.04)
print('saved figs/comb_geometry.png')
