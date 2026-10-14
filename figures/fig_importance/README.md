# fig_importance.png

Feature importance for both classifiers.

    python gen_fig_importance.py     # writes fig_importance.png in place

Reads `../../experiment/bitstream_features_controlled.csv`. The CV seeds and
the final model seed are fixed, so a rerun is deterministic. Use matplotlib
3.10.9 for byte identical output. The script also writes `_also_fig_scores.png`
as a side product.
