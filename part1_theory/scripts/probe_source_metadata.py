#!/usr/bin/env python
"""Per-source metadata for the 50 clips that actually entered the pixel-domain
evaluation.

The 198-sample run did not retain which pool videos it drew, and the companion
script ``describe_corpus.py`` therefore described the *pool* (a random 60 of
1,245) instead. But the rerun archive ``data/pixel_domain_rerun.csv`` records
the ``source`` filename per row, and the corpus directory is still on disk, so
the exact 50 are now identifiable and can be probed directly. This script turns
that into archived numbers, replacing a limitation sentence with measurements.

Outputs (deterministic, re-runnable):
  data/per_source_metadata.csv   one row per source clip
  data/per_source_summary.txt    pooled statistics quoted in the paper

Only container headers are read (``ffmpeg -i`` with no output file), so the run
takes a few seconds and never decodes frames.
"""
from __future__ import annotations

import csv
import os
import re
import statistics
import subprocess

FFMPEG = (r"C:\Users\adhoc\AppData\Local\Programs\Python\Python313"
          r"\Lib\site-packages\imageio_ffmpeg\binaries"
          r"\ffmpeg-win-x86_64-v7.1.exe")
CORPUS = r"C:\baidunetdiskdownload\lightweight\video"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
RERUN_CSV = os.path.join(DATA, "pixel_domain_rerun.csv")
OUT_CSV = os.path.join(DATA, "per_source_metadata.csv")
OUT_TXT = os.path.join(DATA, "per_source_summary.txt")

RE_DUR = re.compile(r"Duration: (\d+):(\d+):(\d+\.?\d*)")
RE_SIZE = re.compile(r"Video: .*? (\d{2,5})x(\d{2,5})")
RE_CODEC = re.compile(r"Video: (\w+)")
RE_FPS = re.compile(r", (\d+\.?\d*) fps")
RE_BR = re.compile(r"bitrate: (\d+) kb/s")


def probe(path: str) -> dict:
    txt = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    row = {"clip": os.path.basename(path)}
    m = RE_DUR.search(txt)
    row["duration_s"] = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                         + float(m.group(3))) if m else -1.0
    m = RE_SIZE.search(txt)
    row["width"], row["height"] = (int(m.group(1)), int(m.group(2))) if m \
        else (-1, -1)
    m = RE_CODEC.search(txt)
    row["codec"] = m.group(1) if m else "?"
    m = RE_FPS.search(txt)
    row["fps"] = float(m.group(1)) if m else -1.0
    m = RE_BR.search(txt)
    row["bitrate_kbps"] = int(m.group(1)) if m else -1
    return row


def main() -> int:
    if not os.path.isdir(CORPUS):
        print("corpus directory missing: %s" % CORPUS)
        return 2
    with open(RERUN_CSV, encoding="utf-8") as fh:
        srcs = sorted({r["source"] for r in csv.DictReader(fh)})
    missing = [s for s in srcs if not os.path.exists(os.path.join(CORPUS, s))]
    if missing:
        print("missing %d/%d source clips" % (len(missing), len(srcs)))
        return 3

    rows = [probe(os.path.join(CORPUS, s)) for s in srcs]
    os.makedirs(DATA, exist_ok=True)
    cols = ["clip", "duration_s", "width", "height", "codec", "fps",
            "bitrate_kbps"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    sizes = sorted({(r["width"], r["height"]) for r in rows})
    codecs = sorted({r["codec"] for r in rows})
    fps_list = sorted({r["fps"] for r in rows})
    dur = [r["duration_s"] for r in rows]
    area = [r["width"] * r["height"] for r in rows]

    lines = [
        "Per-source metadata, the exact 50 clips entering the pixel-domain "
        "evaluation",
        "source list: data/pixel_domain_rerun.csv column 'source'; "
        "probed from %s" % CORPUS,
        "",
        "n sources                 : %d" % len(rows),
        "duration  (s)             : median %.0f, min %.0f, max %.0f"
        % (statistics.median(dur), min(dur), max(dur)),
        "frame area (px)           : median %.0f, min %.0f, max %.0f"
        % (statistics.median(area), min(area), max(area)),
        "distinct resolutions      : %d" % len(sizes),
        "  <=1280x720              : %d"
        % sum(1 for r in rows if r["width"] <= 1280),
        "  1388-1920 wide          : %d"
        % sum(1 for r in rows if 1280 < r["width"] <= 1920),
        "bitrate   (kb/s)          : median %.0f, min %d, max %d"
        % (statistics.median([r["bitrate_kbps"] for r in rows]),
           min(r["bitrate_kbps"] for r in rows),
           max(r["bitrate_kbps"] for r in rows)),
        "fps values                : %s"
        % ", ".join("%g" % f for f in fps_list),
    ]
    for c in codecs:
        lines.append("container codec %-12s: %d"
                     % (c, sum(1 for r in rows if r["codec"] == c)))
    lines += [
        "",
        "NOTE: these replace the earlier pool proxy (60 of 1,245 clips). The "
        "pool proxy",
        "reported a 4096-px upper width that none of the evaluated 50 "
        "reaches, and a",
        "lower HEVC share than the evaluated set actually has.",
    ]
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
