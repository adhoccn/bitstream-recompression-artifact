#!/usr/bin/env python
"""M6(b): describe the source-video pool the 50 sources were drawn from.

The 198-sample pixel-domain run did not retain which 50 pool videos were used
(the per-sample score file was not archived), so we cannot report
resolution/duration for the *subset*.  We therefore report the measured
characteristics of the pool itself and state the gap explicitly, rather than
reconstructing numbers we do not have.
"""
from __future__ import annotations

import os
import random
import re
import subprocess

FFMPEG = (r"C:\Users\adhoc\AppData\Local\Programs\Python\Python313"
          r"\Lib\site-packages\imageio_ffmpeg\binaries"
          r"\ffmpeg-win-x86_64-v7.1.exe")
CORPUS = r"C:\baidunetdiskdownload\lightweight\video"
DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data")
N_SAMPLE = 60
SEED = 20261006


def probe(path: str) -> tuple[float, int, int, str]:
    p = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    txt = p.stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", txt)
    dur = -1.0
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Video: .*?, .*?, .*? (\d{2,5})x(\d{2,5})", txt)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    else:
        m2 = re.search(r"(\d{2,5})x(\d{2,5})", txt)
        w, h = (int(m2.group(1)), int(m2.group(2))) if m2 else (-1, -1)
    m = re.search(r"Video: (\w+)", txt)
    codec = m.group(1) if m else "?"
    return dur, w, h, codec


def main() -> int:
    if not os.path.isdir(CORPUS):
        print("corpus missing")
        return 2
    os.makedirs(DATA, exist_ok=True)
    vids = sorted(v for v in os.listdir(CORPUS) if v.lower().endswith(".mp4"))
    rng = random.Random(SEED)
    picks = rng.sample(vids, min(N_SAMPLE, len(vids)))
    rows = []
    for v in picks:
        try:
            rows.append(probe(os.path.join(CORPUS, v)))
        except Exception:
            pass
    ok = [r for r in rows if r[0] > 0 and r[1] > 0]
    L = []
    L.append("source-pool description (corpus: %s)" % CORPUS)
    L.append("  pool size            : %d clips" % len(vids))
    L.append("  probed sample        : %d clips (seed %d), %d usable"
             % (len(picks), SEED, len(ok)))
    if ok:
        d = sorted(r[0] for r in ok)
        ws = sorted(r[1] for r in ok)
        hs = sorted(r[2] for r in ok)

        def pct(a, p):
            return a[min(len(a) - 1, int(round(p * (len(a) - 1))))]

        L.append("  duration (s)         : min %.1f  p25 %.1f  median %.1f  "
                 "p75 %.1f  max %.1f"
                 % (d[0], pct(d, .25), pct(d, .5), pct(d, .75), d[-1]))
        L.append("  frame width (px)     : min %d  median %d  max %d"
                 % (ws[0], pct(ws, .5), ws[-1]))
        L.append("  frame height (px)    : min %d  median %d  max %d"
                 % (hs[0], pct(hs, .5), hs[-1]))
        from collections import Counter
        c = Counter(r[3] for r in ok)
        L.append("  container video codec: "
                 + ", ".join("%s %d" % kv for kv in c.most_common()))
    L.append("")
    L.append("LIMITATION: the 50 source videos actually used in the 198-sample")
    L.append("run are not identifiable from the retained artifacts, so the above")
    L.append("describes the pool, not the subset.  We state this rather than")
    L.append("reconstruct per-source metadata we do not have.")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(DATA, "corpus_description.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
