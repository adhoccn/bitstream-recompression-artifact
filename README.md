# Artifact: Blind Video Recompression Detection via Bitstream-Level Features

> **Reproducibility artifact for a negative-result paper.**
> The paper reports that bitstream-level (encoding-layer) features **do not** outperform
> the classical pixel-domain baseline once the evaluation protocol removes codec-identity
> leakage. Because the claim is negative, "it reproduces" is the whole point — this
> repository exists so anyone can check it without trusting us.

Submitted to *IEEE Access* (Negative Result). Single-blind, so publishing the artifact
early carries no anonymity risk.

---

## What is in here

```
<repo>/
├── README.md            this file
├── LICENSE              MIT (replace if your institution/funder requires otherwise)
├── .gitignore           Python + LaTeX rules
├── CITATION.cff         machine-readable citation metadata (GitHub renders "Cite this")
├── REPRODUCE.md         full reproduction walkthrough: env, commands, expected numbers
├── requirements.txt     pinned versions actually used (2026-10-12; ffmpeg installed separately)
├── experiment/          the pipeline: sample construction → feature extraction → evaluation
│   ├── prepare_samples.py            build the controlled two-arm corpus (needs source videos + ffmpeg)
│   ├── extract_bitstream_expanded.py 39 bitstream statistics, no pixel decode, no GPU
│   ├── eval_controlled.py            source-aware grouped CV, cross-codec matrix, rate ablation
│   ├── bitstream_features_controlled.csv   the feature table (191 KB) — enables offline reproduction
│   ├── verify_cv_pure.py, _diag_leakage_check.py   protocol guards (source leakage)
│   └── CONTROLLED_RUN.md             field notes from the original run (paths normalized)
└── figures/             one directory per figure, each self-contained
    ├── fig_scores/              gen_fig_scores.py             + rendered PNG + README
    ├── fig_importance/          gen_fig_importance.py         + rendered PNG + README
    └── fig_crosscodec_heatmap/  gen_fig_crosscodec_heatmap.py + rendered PNG + README
```

The CSV is a **feature table, not video content** — it contains no frames, only 39
numeric statistics per clip.

---

## Quick start: verify the headline number without any video

Steps 1–2 of the pipeline need source videos. The feature table ships with the repo,
so the decisive check runs offline:

```bash
pip install -r requirements.txt          # ffmpeg NOT needed for this path
python experiment/eval_controlled.py --features experiment/bitstream_features_controlled.csv
python experiment/eval_controlled.py --self-test    # synthetic data, zero dependencies
```

Expected (measured 2026-10-12, ~38 s on one core):

| Quantity | Value |
|---|---|
| Source-aware grouped CV AUC | **0.6652 ± 0.0123** (paper: 0.665 ± 0.012) |
| Cross-codec off-diagonal | 0.43 – 0.57 |
| NO-RATE ablation vs FULL | 0.6677 vs 0.6652 |

**Read these as "no difference detected", not "a difference was found".** If you get a
markedly stronger number, suspect protocol contamination first — the usual cause is
source leakage (different segments of one original video split across train and test).

See `REPRODUCE.md` §3 for the full table of paper numbers and where each one comes from.

---

## Figures are byte-reproducible

Each `figures/<name>/gen_*.py` regenerates its PNG in place; the PNGs in this repo are
exactly the ones used in the manuscript.

```bash
cd figures/fig_scores && python gen_fig_scores.py     # -> fig_scores.png, 76,497 B
```

Determinism was verified by running every script twice into a temp dir and comparing
md5 (2026-10-11). **matplotlib version matters**: use `matplotlib==3.10.9` for
byte-identical output; other versions change font metrics and therefore the bytes,
but never any number.

---

## Citing

`CITATION.cff` is the authoritative metadata. Two fields are filled after publication:

- `repository_code` — the URL of this repository (fill once it exists)
- `doi` / `identifiers` — the paper DOI and the archived-artifact DOI

Archiving this repository to **Zenodo** or **IEEE DataPort** gives it a citable DOI.
If you use DataPort, **choose Open Access** — otherwise non-member reviewers cannot
download it and the archive is useless.

---

## License

MIT (see `LICENSE`). Code licensing is independent of the paper's licensing: IEEE Access
is fully gold OA with the article text under CC-BY, which does not constrain this repo.
