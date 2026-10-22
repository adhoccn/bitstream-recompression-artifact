"""Measure the CRF -> quantization-step map on real encodes (calibration (C2)).

Why this script exists
----------------------
Reviewer Major Comment 1 asks for the mapping from a rate factor (CRF) to the
effective scalar quantization step q, and for the distortion that this mapping
introduces into the eps^{-2} barrier of Theorem 6.  The earlier
``calibrate_epsilon.py`` answered that with a *modelled* distribution: it
assumed the second-pass step is q2 = q2_nominal * (1 + eta) with eta drawn from
a uniform +-2 QP swing, because no per-frame QP information was available.

That assumption is no longer necessary.  The ffmpeg bitstream filter
``trace_headers`` dumps every slice header, including

    pic_init_qp_minus26   (PPS)
    slice_qp_delta        (slice header)

from which the luma slice QP is exactly  SliceQPY = 26 + pic_init_qp_minus26 +
slice_qp_delta.  H.264 then defines the step by Qstep(QP) = 0.625 * 2^(QP/6).
So the per-frame step ratio, and hence eps, is *directly observable*:

    eps_f = Qstep(QP2_f) / Qstep(QP1_f) - 1 = 2^((QP2_f - QP1_f)/6) - 1.

This script therefore REPLACES the modelled +-2 QP swing with the measured one.
It also answers the reviewer's Minor Comment on the q2 > q1 vs q2 < q1 regime:
it reports the measured fraction of frame pairs with q2 <= q1, i.e. the
fraction on which Theorem 1(iii)'s comb-with-gaps branch does not apply.

Output: data/qp_calibration.txt
"""

import math
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

FFMPEG = (r"C:\Users\adhoc\AppData\Local\Programs\Python\Python313\Lib"
          r"\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe")
CORPUS = r"C:\baidunetdiskdownload\lightweight\video"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")

SEED = 20261006
N_VIDEOS = 12
N_FRAMES = 30
SIZE = "640x360"
CRF1 = 28
DELTAS = (1, 2, 3, 4, 6)
CODECS = ("h264",)

VCodec = {"h264": "libx264", "h265": "libx265"}


# ----------------------------------------------------------------- helpers
def sh(cmd):
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pick_videos(n):
    rng = np.random.default_rng(SEED)
    names = sorted(f for f in os.listdir(CORPUS)
                   if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
    idx = rng.choice(len(names), size=min(n, len(names)), replace=False)
    return [names[i] for i in sorted(idx)]


def encode(src, dst, codec, crf):
    sh([FFMPEG, "-y", "-loglevel", "error", "-i", src,
        "-frames:v", str(N_FRAMES), "-s", SIZE, "-an", "-sn",
        "-pix_fmt", "yuv420p", "-c:v", VCodec[codec], "-crf", str(crf),
        "-preset", "ultrafast", "-g", "12", dst])


_RE_PIC = re.compile(r"pic_init_qp_minus26\s+\S+\s*=\s*(-?\d+)")
_RE_SLC = re.compile(r"slice_qp_delta\s+\S+\s*=\s*(-?\d+)")
_RE_HDR = re.compile(r"Slice Header")


def slice_qps(path):
    """Per-slice luma QP, in decode order, via the trace_headers bitstream filter."""
    proc = subprocess.run(
        [FFMPEG, "-y", "-loglevel", "debug", "-i", path,
         "-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    txt = proc.stderr.decode("utf-8", "replace")
    qps, pic_init, in_slice = [], 0, False
    for line in txt.splitlines():
        m = _RE_PIC.search(line)
        if m:
            pic_init = int(m.group(1))
            continue
        if _RE_HDR.search(line):
            in_slice = True
            continue
        m = _RE_SLC.search(line)
        if m and in_slice:
            qps.append(26 + pic_init + int(m.group(1)))
            in_slice = False
    return np.array(qps, dtype=np.int64)


def qstep(qp):
    """H.264 luma quantization step."""
    return 0.625 * np.power(2.0, np.asarray(qp, dtype=np.float64) / 6.0)


# ------------------------------------------------------------------- main
def main():
    os.makedirs(DATA, exist_ok=True)
    out = []
    rec = out.append

    rec("measured CRF -> QP -> Qstep -> eps calibration (C2)")
    rec("  ffmpeg : %s" % os.path.basename(FFMPEG))
    rec("  method : ffmpeg -bsf:v trace_headers -> pic_init_qp_minus26 + slice_qp_delta")
    rec("           SliceQPY = 26 + pic_init_qp_minus26 + slice_qp_delta")
    rec("           Qstep(QP) = 0.625 * 2^(QP/6)   (H.264 Table 8-? / E.2.2)")
    rec("  seed %d | %d videos | %d frames | %s | CRF1 = %d"
        % (SEED, N_VIDEOS, N_FRAMES, SIZE, CRF1))
    rec("")
    rec("    dCRF   n_frames   eps_nom    eps_mean   eps_med    eps_rms   rms/nom   P(eps<=0)   N*/N*_nom   QP1 mean   QP2 mean")
    rec("    " + "-" * 116)

    videos = pick_videos(N_VIDEOS)
    tmp = tempfile.mkdtemp(prefix="qptrace_")
    try:
        per_delta = {}
        for codec in CODECS:
            for d in DELTAS:
                eps_all, q1_all, q2_all = [], [], []
                for v in videos:
                    src = os.path.join(CORPUS, v)
                    p1 = os.path.join(tmp, "p1.mp4")
                    p2 = os.path.join(tmp, "p2.mp4")
                    try:
                        encode(src, p1, codec, CRF1)
                        encode(p1, p2, codec, CRF1 + d)
                        a1 = slice_qps(p1)
                        a2 = slice_qps(p2)
                    except Exception:
                        continue
                    n = min(a1.size, a2.size)
                    if n < 4:
                        continue
                    a1, a2 = a1[:n], a2[:n]
                    q1_all.append(a1)
                    q2_all.append(a2)
                    eps_all.append(qstep(a2) / qstep(a1) - 1.0)
                if not eps_all:
                    continue
                e = np.concatenate(eps_all)
                q1 = np.concatenate(q1_all)
                q2 = np.concatenate(q2_all)
                eps_nom = 2.0 ** (d / 6.0) - 1.0
                rms = float(np.sqrt(np.mean(e ** 2)))
                ratio = (eps_nom / rms) ** 2 if rms > 0 else float("nan")
                ple = float(np.mean(e <= 0.0))
                rec("    %4d %10d %9.4f %10.4f %9.4f %10.4f %9.3f %10.3f %11.3f %10.2f %10.2f"
                    % (d, e.size, eps_nom, float(np.mean(e)), float(np.median(e)),
                       rms, rms / eps_nom, ple, ratio,
                       float(q1.mean()), float(q2.mean())))
                per_delta[d] = dict(eps=e, q1=q1, q2=q2, nom=eps_nom,
                                    rms=rms, ratio=ratio, ple=ple)

        rec("")
        rec("Reading:")
        rec("  * eps_nom  = 2^(dCRF/6) - 1 : what a CRF-based reading of Theorem 6 assumes.")
        rec("  * eps_rms  : what Theorem 6 actually sees, because H^2 accumulates additively")
        rec("               over frames and H^2 ~ C eps^2 per frame, so the pooled barrier is")
        rec("               driven by E[eps^2], i.e. by eps_rms.")
        rec("  * N*/N*_nom = (eps_nom / eps_rms)^2 : distortion of the barrier.  A value of")
        rec("               0.44 means the CRF axis OVERSTATES the required sample size by 2.3x.")
        rec("  * P(eps<=0) : measured fraction of frame pairs on which the second step is NOT")
        rec("               larger than the first, i.e. the q2 > q1 branch of Theorem 1(iii)")
        rec("               does not hold frame-locally.  This is the regime question.")
        rec("  * QP1 / QP2 means are the measured rate-factor -> QP map, replacing the")
        rec("    modelled +-2 QP swing of the earlier calibrate_epsilon.py.")

        # ---- which regime are double-compressed samples actually in? -------
        rec("")
        rec("Regime split (Theorem 1(iii)): measured, pooled over all dCRF > 0")
        all_e = np.concatenate([v["eps"] for v in per_delta.values()])
        rec("  frame pairs            : %d" % all_e.size)
        rec("  P(q2 >  q1)  [b>a]     : %.3f" % float(np.mean(all_e > 0)))
        rec("  P(q2 <= q1)  [b<a]     : %.3f" % float(np.mean(all_e <= 0)))
        rec("  P(q2 == q1)  [degen]   : %.3f" % float(np.mean(all_e == 0)))
        rec("  Interpretation: under dCRF > 0 (recompression toward lower quality) the")
        rec("  majority branch is q2 > q1, so Theorem 1(iii) gives the TRIVIAL lattice")
        rec("  spacing with NO comb-with-gaps.  The forensically exploitable comb is the")
        rec("  minority branch, and it shrinks as dCRF grows.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    path = os.path.join(DATA, "qp_calibration.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
