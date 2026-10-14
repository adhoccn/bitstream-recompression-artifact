# Reproduction

## 1. Environment

Python 3.13, ffmpeg on PATH, and the packages in `requirements.txt`:

    pip install -r requirements.txt

ffmpeg is a separate install, not a Python package. Feature extraction reads
the container with PyAV, so a working ffmpeg build matters.

## 2. Pipeline

Step 1 needs source videos. Steps 2 and 3 need only the index CSV and the
feature table.

    # 1. build the two-arm corpus
    python experiment/prepare_samples.py --video-dir /path/to/videos --max 20

    # 2. extract bitstream features (no pixel decoding, no GPU)
    python experiment/extract_bitstream_expanded.py \
        --index  data_controlled/sample_index.csv \
        --output results/bitstream_features_controlled.csv

    # 3. evaluate
    python experiment/eval_controlled.py --features results/bitstream_features_controlled.csv

`experiment/bitstream_features_controlled.csv` ships with the repository, so
the evaluation also runs with no video at all:

    python experiment/eval_controlled.py --features experiment/bitstream_features_controlled.csv
    python experiment/eval_controlled.py --self-test

## 3. What the evaluation prints

- source-aware grouped CV: AUC, EER, TDR at two false-positive levels, mean
  and confidence interval over seeds
- a cross-codec train and test matrix
- a rate-confound ablation, full features against rate features removed

`--model xgb` or `--model lr` selects the classifier. `--max-delta 3`
restricts the run to the near-parameter subset.

## 4. Protocol guard

`experiment/verify_cv_pure.py` checks that no clip appears in both a training
and a test fold. Run it after producing a feature table:

    python experiment/verify_cv_pure.py

## 5. Figures

    cd figures/fig_scores && python gen_fig_scores.py

Each generator writes its PNG in place and prints the numbers it plotted. Use
matplotlib 3.10.9 for byte identical output.

## 6. Paths

All defaults are relative to the repository root. Override them with CLI flags
(`--video-dir`, `--data-dir`, `--ffmpeg`) or with the environment variables
`GCRR_VIDEO_DIR`, `GCRR_DATA_DIR`, `GCRR_FFMPEG`.
