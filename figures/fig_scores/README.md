# fig_scores.png

Score distributions of the two arms under source-aware cross-validation.

    python gen_fig_scores.py     # writes fig_scores.png in place

Reads `../../experiment/bitstream_features_controlled.csv`. The CV seeds and
the final model seed are fixed, so a rerun is deterministic. Use matplotlib
3.10.9 for byte identical output. The script also writes
`_also_fig_importance.png` as a side product.
