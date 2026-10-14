import sys
from pathlib import Path

# daimon_runtime (old env helper) no longer exists; set the publication font
# directly: Times New Roman for all in-figure text, STIX for math.
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif',
                   'serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

import numpy as np
import matplotlib.pyplot as plt

codecs = ['H.264', 'H.265', 'VP9']
# mean values
M = np.array([
    [0.701, 0.589, 0.560],
    [0.530, 0.631, 0.455],
    [0.508, 0.483, 0.619],
])
CI = np.array([
    [0.022, 0.012, 0.008],
    [0.014, 0.020, 0.011],
    [0.010, 0.021, 0.027],
])

fig, ax = plt.subplots(figsize=(5.2, 4.2))
im = ax.imshow(M, cmap='YlOrRd', vmin=0.40, vmax=0.75)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(codecs); ax.set_yticklabels(codecs)
ax.set_xlabel('Test Codec'); ax.set_ylabel('Train Codec')
for i in range(3):
    for j in range(3):
        ax.text(j, i - 0.08, f"{M[i, j]:.3f}", ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if M[i, j] > 0.62 else 'black')
        ax.text(j, i + 0.22, f"±{CI[i, j]:.3f}", ha='center', va='center',
                fontsize=7.5, color='white' if M[i, j] > 0.62 else 'black')
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label('AUC-ROC')
fig.text(0.5, 0.01,
         'Diagonal: leave-source-out CV (10 seeds × 5 folds). '
         'Off-diagonal: mean over 10 seeds (subsampled XGBoost). All ± 95% CI.',
         ha='center', fontsize=7, color='0.35')
fig.tight_layout(rect=[0, 0.04, 1, 1])

for out in [str(Path(__file__).resolve().parent / 'fig_crosscodec_heatmap.png')]:
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('saved', out)
