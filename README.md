# bitstream-recompression-artifact

Artifact for *Blind Video Recompression Detection via Bitstream-Level
Encoding-Layer Features: A Controlled Protocol and a Negative Result*.

Code and data for extracting bitstream-level (encoding-layer) features from
MP4 files and evaluating them under a controlled two-arm protocol. Both arms
are encoded with the same codec and the same first-pass quality, so the single
versus double comparison is not confounded by codec identity. The result is
negative: AUC 0.665 +/- 0.012 under source-aware grouped cross-validation.

## Layout

    experiment/    pipeline scripts plus the extracted feature table
    figures/       one directory per figure, each with its generator

## Requirements

Python 3.13, ffmpeg on PATH, and the packages in `requirements.txt`:

    pip install -r requirements.txt

ffmpeg is a separate install, not a Python package.

## Pipeline

    # 1. build the two-arm corpus (needs source videos and ffmpeg)
    python experiment/prepare_samples.py --video-dir /path/to/videos --max 20

    # 2. extract 39 bitstream statistics per clip (no pixel decoding, no GPU)
    python experiment/extract_bitstream_expanded.py \
        --index  data_controlled/sample_index.csv \
        --output results/bitstream_features_controlled.csv

    # 3. evaluate: source-aware grouped CV, cross-codec matrix, rate ablation
    python experiment/eval_controlled.py --features results/bitstream_features_controlled.csv

The repository ships `experiment/bitstream_features_controlled.csv`, so the
evaluation also runs with no video at all:

    python experiment/eval_controlled.py --features experiment/bitstream_features_controlled.csv
    python experiment/eval_controlled.py --self-test

## Figures

    cd figures/fig_scores && python gen_fig_scores.py

Each generator writes its PNG in place and prints the numbers it plotted.
Use matplotlib 3.10.9 for byte identical output; other versions change font
metrics and therefore the file bytes, but not any number.

## Paths

All defaults are relative to the repository root. Override them with CLI flags
(`--video-dir`, `--data-dir`, `--ffmpeg`) or with the environment variables
`GCRR_VIDEO_DIR`, `GCRR_DATA_DIR`, `GCRR_FFMPEG`.

## Part I theory artifacts

This repository additionally archives the theory-verification code behind
**Part I** of the companion series, *Signature Existence and Optimal
Multi-Stage Fusion for Blind Video Recompression Detection*, under
`part1_theory/`:

    part1_theory/scripts/     theory validation scripts cited in Part I
    part1_theory/data/        stored CSV/TXT outputs behind each table
    part1_theory/figures/     one directory per figure: PNG + its generator
    part1_theory/FIGURE_TABLE_MAP.md   which script draws which figure

Every script prints the numbers it plots rather than only drawing them, so each
quantity quoted in the manuscript can be checked from stdout. Figures were
produced on Python 3.13 with matplotlib 3.10.9; reproduce with

    python part1_theory/scripts/validate_signature.py
    python part1_theory/scripts/measure_deblock_sigma.py
    python part1_theory/scripts/make_figures.py

## License

MIT, see `LICENSE`.
