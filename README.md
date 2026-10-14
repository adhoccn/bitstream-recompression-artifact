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

## License

MIT, see `LICENSE`.
