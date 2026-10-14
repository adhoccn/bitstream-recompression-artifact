import csv
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# NOTE: the original script's lost helper (daimon_runtime.setup_plot) rendered
# these figures on seaborn's "darkgrid" background; kept for visual parity.
sns.set_theme(style="darkgrid", context="paper", font_scale=1.0,
              palette="colorblind")
# daimon_runtime (old env helper) no longer exists; set the publication font
# directly AFTER seaborn's set_theme (which would otherwise override it):
# Times New Roman for all in-figure text, STIX for math.
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif',
                   'serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# ============================================================
# Load data
# ============================================================
# feature table: figures/<name>/ -> repo root -> experiment/
_CSV = (Path(__file__).resolve().parent.parent.parent
        / 'experiment' / 'bitstream_features_controlled.csv')
if not _CSV.exists():
    raise FileNotFoundError(
        f'feature table not found: {_CSV}\n'
        'this repository ships it; if missing, regenerate it with '
        'experiment/extract_bitstream_expanded.py.')
CSV_PATH = str(_CSV)

df = pd.read_csv(CSV_PATH)

# Identify numeric feature columns
exclude = {'duration_s', 'width', 'height', 'video_file', 'label', 'codec',
           'crf', 'sample_path', 'codec_name', 'profile', 'pixel_format',
           'error', 'n_frames_decoded', 'other_ratio', 'level',
           'q1', 'q2', 'delta_qp'}
feature_cols = [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

X = df[feature_cols].fillna(0).values.astype(np.float64)
y = (df['label'] == 'double').astype(int).values
source_ids = df['video_file'].values

print(f"Samples: {len(y)}, single={sum(y==0)}, double={sum(y==1)}")
print(f"Features: {len(feature_cols)}")
print(f"Sources: {len(np.unique(source_ids))}")

# ============================================================
# Figure 1: Score distributions (source-aware CV)
# ============================================================
N_SEEDS = 10
N_FOLDS = 5

all_scores = np.zeros(len(y))
all_counts = np.zeros(len(y))

for seed in range(N_SEEDS):
    rng = np.random.RandomState(seed)
    unique_sources = np.unique(source_ids)
    shuffled = rng.permutation(unique_sources)
    fold_size = len(shuffled) // N_FOLDS

    for fold in range(N_FOLDS):
        test_srcs = set(shuffled[fold * fold_size:(fold + 1) * fold_size])
        mask_tr = np.array([s not in test_srcs for s in source_ids])
        mask_te = np.array([s in test_srcs for s in source_ids])

        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed, eval_metric='logloss', verbosity=0
        )
        model.fit(X[mask_tr], y[mask_tr])
        prob = model.predict_proba(X[mask_te])[:, 1]
        all_scores[mask_te] += prob
        all_counts[mask_te] += 1

# Average scores across seeds (each sample gets ~2 predictions on average)
avg_scores = np.divide(all_scores, all_counts, out=np.zeros_like(all_scores), where=all_counts>0)
scores_single = avg_scores[y == 0]
scores_double = avg_scores[y == 1]

# ---- Plot ----
CB_BLUE = sns.color_palette("colorblind")[0]   # single
CB_ORANGE = sns.color_palette("colorblind")[1] # double

fig1, ax1 = plt.subplots(figsize=(6.0, 3.8), dpi=300)

bins = np.linspace(0, 1, 31)
ax1.hist(scores_single, bins=bins, color=CB_BLUE, alpha=0.65,
         label='single (y=0)', density=True, edgecolor='white', linewidth=0.5)
ax1.hist(scores_double, bins=bins, color=CB_ORANGE, alpha=0.65,
         label='double (y=1)', density=True, edgecolor='white', linewidth=0.5)

ax1.set_xlabel('XGBoost double-compression score (out-of-fold)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Source-Aware CV: Score Distributions', fontsize=13, weight='bold', pad=10)
ax1.legend(loc='upper left', fontsize=10, frameon=True, fancybox=False, edgecolor='gray')
ax1.set_xlim(-0.02, 1.02)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=10)

plt.tight_layout()
out1 = str(Path(__file__).resolve().parent / '_also_fig_scores.png')
fig1.savefig(out1, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig1)
print(f'[SAVED] {out1}')

# ============================================================
# Figure 2: Feature importance (XGB gain + LR |coef|)
# ============================================================
# Train final models on full data
xgb_final = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    random_state=42, eval_metric='logloss', verbosity=0
)
xgb_final.fit(X, y)

# XGB gain importance
xgb_imp = xgb_final.feature_importances_
xgb_top_idx = np.argsort(xgb_imp)[::-1][:10]
xgb_names = [feature_cols[i] for i in xgb_top_idx]
xgb_vals = xgb_imp[xgb_top_idx]

# LR standardized coefficients
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
lr_final = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_final.fit(X_s, y)
lr_coef = np.abs(lr_final.coef_[0])
lr_top_idx = np.argsort(lr_coef)[::-1][:10]
lr_names = [feature_cols[i] for i in lr_top_idx]
lr_vals = lr_coef[lr_top_idx]

# ---- Plot ----
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=300,
                                   gridspec_kw={'wspace': 0.35})

# Left: XGBoost
y_pos = np.arange(len(xgb_names))[::-1]
bars1 = ax2a.barh(y_pos, xgb_vals, color=CB_BLUE, height=0.65, edgecolor='white')
ax2a.set_yticks(y_pos)
ax2a.set_yticklabels(xgb_names, fontsize=10)
ax2a.set_xlabel('Normalized Gain', fontsize=11)
ax2a.set_title('XGBoost (Top-10)', fontsize=12, weight='bold', pad=8)
ax2a.set_xlim(0, max(xgb_vals) * 1.15)
ax2a.spines['top'].set_visible(False)
ax2a.spines['right'].set_visible(False)
ax2a.tick_params(labelsize=9)

# Right: Logistic Regression
y_pos2 = np.arange(len(lr_names))[::-1]
bars2 = ax2b.barh(y_pos2, lr_vals, color=sns.color_palette("colorblind")[2],
                  height=0.65, edgecolor='white')
ax2b.set_yticks(y_pos2)
ax2b.set_yticklabels(lr_names, fontsize=10)
ax2b.set_xlabel('|Standardized Weight|', fontsize=11)
ax2b.set_title('Logistic Regression (Top-10)', fontsize=12, weight='bold', pad=8)
ax2b.set_xlim(0, max(lr_vals) * 1.15)
ax2b.spines['top'].set_visible(False)
ax2b.spines['right'].set_visible(False)
ax2b.tick_params(labelsize=9)

# Shared suptitle
fig2.suptitle('Feature Importance', fontsize=13, weight='bold', y=1.02)

plt.tight_layout()
out2 = str(Path(__file__).resolve().parent / 'fig_importance.png')
fig2.savefig(out2, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig2)
print(f'[SAVED] {out2}')

print('\n=== Summary ===')
print(f'fig_scores.png:     {scores_single.mean():.3f} (single mean) vs {scores_double.mean():.3f} (double mean)')
print(f'fig_importance.png: XGB top = {xgb_names[0]} ({xgb_vals[0]:.3f}), LR top = {lr_names[0]} ({lr_vals[0]:.3f})')

"""Regenerate fig_importance.png from the bitstream feature table."""

