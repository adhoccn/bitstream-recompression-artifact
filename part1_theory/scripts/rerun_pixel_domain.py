"""Rebuild the pixel-domain instantiation and ARCHIVE THE PER-SAMPLE SCORES.

Why this script exists
----------------------
Reviewer Major Comment 6(a)(c) and Minor Comment 10 ask for cluster-adjusted
inference, for repeated-measurement variability on the fitted exponents, and for
the sign and magnitude of the logistic-regression coefficients.  None of these
can be produced from the original 198-sample run because that run archived only
aggregate statistics: the per-sample score vector was never written to disk, so
the intra-cluster correlation rho cannot be estimated, no cluster bootstrap can
be run, and no coefficient vector can be reported.

This script therefore RERUNS the pixel-domain instantiation under a fully
specified, reproducible protocol and writes the per-sample score vector to
``data/pixel_domain_rerun.csv``.  It then recomputes everything that was asked
for from that vector.

Protocol
--------
50 source clips (seed 20261006).  Per source:
  * pass 1 : encode the first 10 s at CRF 28 (libx264, ultrafast)  -> the
             SINGLE-compressed sample (H0), n = 50;
  * pass 2 : re-encode pass 1 at CRF 31 (Delta CRF = 3, the near-parameter
             regime the paper discusses) in H.264, H.265 and VP9 -> three
             DOUBLE-compressed samples (H1), n = 150.
Total n = 200 (the original run had 198: 50 + 148).

Features (all from DECODED pixels, i.e. the classical pipeline)
--------------------------------------------------------------
Stage 1  temporal inconsistency, 5 features, fixed equal weights
Stage 2  DCT double-quantization periodicity, fundamental-to-DC ratio R_Delta
Stage 3  JPEG-proxy regional distance, MEAN over a 4x4 grid (0.6*rd + 0.4*rb)
Stage 4  the same regional score aggregated by MAX over the grid

Only numpy is required; the logistic regression is an explicit IRLS fit so that
its coefficients are reproducible without sklearn.
"""

import csv
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np

FFMPEG = (r"C:\Users\adhoc\AppData\Local\Programs\Python\Python313\Lib"
          r"\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe")
CORPUS = r"C:\baidunetdiskdownload\lightweight\video"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")

SEED = 20261006
N_SOURCES = int(os.environ.get("PD_N_SOURCES", "50"))   # smoke-test override
N_FRAMES = 30            # sampled at 3 fps over the first 10 s
N_JPEG = 6               # frames used for the JPEG-proxy stages
W, H = 640, 360
CRF1 = 28
DCRF = 3
CODECS = ("h264", "h265", "vp9")
VCODEC = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9"}
GRID = 4

# orthonormal 8x8 DCT
_k = np.arange(8)
_D8 = np.cos(np.pi * (_k[:, None] + 0.5) * _k[None, :] / 8.0) * np.sqrt(2.0 / 8.0)
_D8[0, :] = 1.0 / np.sqrt(8.0)


# ------------------------------------------------------------------ ffmpeg
def sh(cmd):
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_raw(src, dst):
    """First 10 s, gray, 640x360, 3 fps -> N_FRAMES raw frames."""
    sh([FFMPEG, "-y", "-loglevel", "error", "-i", src,
        "-t", "10", "-an", "-sn", "-vf",
        "scale=%d:%d,fps=3" % (W, H), "-pix_fmt", "gray",
        "-f", "rawvideo", dst])


def encode_raw(raw_in, dst, codec, crf):
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", "%dx%d" % (W, H),
           "-r", "3", "-i", raw_in,
           "-an", "-sn", "-c:v", VCODEC[codec]]
    if codec == "vp9":
        cmd += ["-crf", str(crf), "-b:v", "0", "-deadline", "realtime",
                "-cpu-used", "8"]
    else:
        cmd += ["-crf", str(crf), "-preset", "ultrafast"]
    cmd += ["-g", "12", dst]
    sh(cmd)


def decode_raw(src, dst):
    sh([FFMPEG, "-y", "-loglevel", "error", "-i", src,
        "-f", "rawvideo", "-pix_fmt", "gray", dst])


def jpeg_roundtrip(raw_in, raw_out, n_frames, q=5):
    """JPEG-proxy: encode the first n_frames raw frames to MJPEG and back."""
    with open(raw_in, "rb") as fi:
        buf = fi.read(n_frames * W * H)
    tmp_in = raw_in + ".jin"
    with open(tmp_in, "wb") as fo:
        fo.write(buf)
    sh([FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", "%dx%d" % (W, H),
        "-r", "3", "-i", tmp_in, "-c:v", "mjpeg", "-q:v", str(q),
        "-f", "rawvideo", "-pix_fmt", "gray", raw_out])
    os.remove(tmp_in)


def read_raw(path, n=N_FRAMES):
    a = np.fromfile(path, dtype=np.uint8)
    k = min(n, a.size // (W * H))
    if k == 0:
        return np.zeros((1, H, W))
    out = a[: k * W * H].reshape(k, H, W).astype(np.float64)
    if k < n:
        out = np.concatenate([out, np.repeat(out[-1:], n - k, axis=0)], axis=0)
    return out


def mean_qp(path):
    """Mean luma slice QP via trace_headers; NaN if unavailable.

    Returns NaN rather than 0 when the codec exposes no slice QP (VP9 stores its
    quantizer in a different syntax element), so that an unavailable trace can
    never be silently read as "QP = 0" and corrupt the branch label.
    """
    try:
        p = subprocess.run([FFMPEG, "-y", "-loglevel", "debug", "-i", path,
                            "-c", "copy", "-bsf:v", "trace_headers",
                            "-f", "null", "-"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=120)
        txt = p.stderr.decode("utf-8", "replace")
    except Exception:
        return float("nan")
    import re
    # H.264: pic_init_qp_minus26 (PPS).  HEVC: init_qp_minus26 (PPS).
    rp = re.compile(r"(?:pic_)?init_qp_minus26\s+\S+\s*=\s*(-?\d+)")
    rs = re.compile(r"slice_qp_delta\s+\S+\s*=\s*(-?\d+)")
    qps, pic, insl = [], 0, False
    for line in txt.splitlines():
        m = rp.search(line)
        if m:
            pic, insl = int(m.group(1)), False
            continue
        if "Slice Header" in line or "slice_segment_header" in line:
            insl = True
            continue
        m = rs.search(line)
        if m and insl:
            qps.append(26 + pic + int(m.group(1)))
            insl = False
    return float(np.mean(qps)) if qps else float("nan")


# ---------------------------------------------------------------- features
def block_ac(frame):
    h, w = frame.shape
    h -= h % 8
    w -= w % 8
    f = frame[:h, :w]
    blk = (f.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
            .reshape(-1, 8, 8))
    coef = _D8 @ blk @ _D8.T
    ac = coef.copy()
    ac[:, 0, 0] = 0.0
    return coef, ac


def stage1(frames):
    """Temporal inconsistency: 5 features on the per-frame AC-energy series."""
    e = np.array([np.abs(block_ac(f)[1]).mean() for f in frames],
                 dtype=np.float64)
    if e.size < 8 or e.mean() <= 0:
        return np.zeros(5)
    x = e - e.mean()
    spec = np.abs(np.fft.rfft(x))[1:16]
    med = float(np.median(spec)) + 1e-12
    f1 = float(spec.max() / med)
    f2 = float(np.sum(spec > 3 * med))
    xx = x - x.mean()
    denom = float(np.dot(xx, xx)) + 1e-12
    def acf(lag):
        if lag >= xx.size:
            return 0.0
        return abs(float(np.dot(xx[:-lag], xx[lag:])) / denom)
    f3, f4 = acf(12), acf(24)
    f5 = float(e.std() / (e.mean() + 1e-12))
    return np.array([f1, f2, f3, f4, f5])


C1 = np.array([3.0, 3.0, 1.0, 1.0, 1.0])


def stage2(frames):
    """DCT double-quantization periodicity: fundamental-to-DC ratio R_Delta."""
    ac = np.concatenate([block_ac(f)[1].ravel() for f in frames])
    ac = ac[np.abs(ac) > 1e-6]
    if ac.size < 1000:
        return 0.0
    lim = float(np.percentile(np.abs(ac), 99.0)) * 1.2 + 1e-6
    L = 2048
    idx = np.clip(np.floor((ac + lim) / (2 * lim) * L).astype(np.int64), 0, L - 1)
    h = np.bincount(idx, minlength=L).astype(np.float64)
    H = np.abs(np.fft.rfft(h - h.mean()))
    if H[0] <= 0:
        return 0.0
    best = 0.0
    for d in range(2, 9):
        j = L // d
        if 0 < j < H.size:
            best = max(best, float(H[j] / H[0]))
    return best


def regional_scores(frames, jframes):
    """JPEG-proxy regional distance on a GRID x GRID grid: 0.6*rd + 0.4*rb."""
    n = min(frames.shape[0], jframes.shape[0])
    gh, gw = H // GRID, W // GRID
    out = np.zeros((n, GRID, GRID))
    for t in range(n):
        a, b = frames[t], jframes[t]
        d = np.abs(a - b) / 255.0
        # blockiness: excess difference across 8-px horizontal boundaries
        dh = np.abs(np.diff(a, axis=1))
        bd = dh[:, 7::8].mean() if dh[:, 7::8].size else 0.0
        base = dh.mean() + 1e-9
        rb = max(0.0, (bd - dh.mean()) / base)
        for i in range(GRID):
            for j in range(GRID):
                rd = float(d[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].mean())
                out[t, i, j] = 0.6 * rd + 0.4 * rb
    return out


# ------------------------------------------------------------- statistics
def auc_mw(y, s):
    """AUC, Hanley-McNeil SE, Mann-Whitney z and two-sided p."""
    y = np.asarray(y)
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n0 == 0 or n1 == 0:
        return (float("nan"),) * 4
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), dtype=np.float64)
    ss = np.asarray(s)[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    U = float(np.sum(r[y == 1]) - n1 * (n1 + 1) / 2.0)
    A = U / (n0 * n1)
    Q1 = A / (2 - A)
    Q2 = 2 * A * A / (1 + A)
    se = math.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A)
                    + (n0 - 1) * (Q2 - A * A)) / (n0 * n1))
    mu = n0 * n1 / 2.0
    sd = math.sqrt(n0 * n1 * (n0 + n1 + 1) / 12.0)
    z = (U - mu) / sd
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return A, se, z, p


def cluster_bootstrap_auc(y, s, groups, rng, B=2000):
    """Resample CLUSTERS (source videos) with replacement."""
    g = np.unique(groups)
    vals = []
    for _ in range(B):
        pick = rng.choice(g, size=g.size, replace=True)
        idx = np.concatenate([np.flatnonzero(groups == c) for c in pick])
        yy, ss = np.asarray(y)[idx], np.asarray(s)[idx]
        if np.sum(yy == 0) == 0 or np.sum(yy == 1) == 0:
            continue
        vals.append(auc_mw(yy, ss)[0])
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def icc_rho(resid, groups):
    """One-way ANOVA intraclass correlation on residuals."""
    g = np.unique(groups)
    if g.size < 2:
        return float("nan")
    gm = np.array([resid[groups == c].mean() for c in g])
    nbar = resid.size / g.size
    msb = nbar * np.var(gm, ddof=1)
    msw = np.mean([np.var(resid[groups == c], ddof=1)
                   for c in g if np.sum(groups == c) > 1])
    if msb + (nbar - 1) * msw <= 0:
        return float("nan")
    return float((msb - msw) / (msb + (nbar - 1) * msw))


def logistic_irls(X, y, iters=60, ridge=1e-6):
    """Unregularized logistic regression by IRLS; returns coefficients."""
    n, p = X.shape
    Xd = np.concatenate([np.ones((n, 1)), X], axis=1)
    w = np.zeros(p + 1)
    for _ in range(iters):
        eta = np.clip(Xd @ w, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(pr * (1 - pr), 1e-8, None)
        H = Xd.T @ (Xd * W[:, None]) + ridge * np.eye(p + 1)
        g = Xd.T @ (y - pr)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w          # [intercept, coef_1, ...]


# -------------------------------------------------------------------- main
def main():
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(SEED)
    names = sorted(f for f in os.listdir(CORPUS)
                   if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
    picks = [names[i] for i in
             sorted(rng.choice(len(names), size=min(N_SOURCES, len(names)),
                               replace=False))]

    tmp = tempfile.mkdtemp(prefix="pixdom_")
    rows = []
    try:
        for si, v in enumerate(picks):
            src = os.path.join(CORPUS, v)
            raw0 = os.path.join(tmp, "s.raw")
            p1 = os.path.join(tmp, "p1.mp4")
            d1 = os.path.join(tmp, "d1.raw")
            try:
                extract_raw(src, raw0)
                encode_raw(raw0, p1, "h264", CRF1)
                decode_raw(p1, d1)
                f1 = read_raw(d1)
                qp1 = mean_qp(p1)
            except Exception:
                continue
            if f1.shape[0] < 8:
                continue
            jr = os.path.join(tmp, "j1.raw")
            try:
                jpeg_roundtrip(d1, jr, N_JPEG)
                j1 = read_raw(jr, N_JPEG)
            except Exception:
                j1 = f1[:N_JPEG]
            reg1 = regional_scores(f1[:N_JPEG], j1)
            rows.append(dict(source=v, codec="h264(single)", label=0,
                             S1=float(np.mean(np.minimum(stage1(f1) / C1, 1.0))),
                             S2=float(stage2(f1)),
                             S3=float(reg1.mean()), S4=float(reg1.max()),
                             qp1=qp1, qp2=float("nan")))   # no second pass
            for cdec in CODECS:
                p2 = os.path.join(tmp, "p2.mp4")
                d2 = os.path.join(tmp, "d2.raw")
                try:
                    encode_raw(d1, p2, cdec, CRF1 + DCRF)
                    decode_raw(p2, d2)
                    f2 = read_raw(d2)
                    qp2 = mean_qp(p2)
                except Exception:
                    continue
                jr2 = os.path.join(tmp, "j2.raw")
                try:
                    jpeg_roundtrip(d2, jr2, N_JPEG)
                    j2 = read_raw(jr2, N_JPEG)
                except Exception:
                    j2 = f2[:N_JPEG]
                reg2 = regional_scores(f2[:N_JPEG], j2)
                rows.append(dict(source=v, codec=cdec, label=1,
                                 S1=float(np.mean(np.minimum(stage1(f2) / C1, 1.0))),
                                 S2=float(stage2(f2)),
                                 S3=float(reg2.mean()), S4=float(reg2.max()),
                                 qp1=qp1, qp2=qp2))
            print("  [%2d/%d] %s" % (si + 1, len(picks), v), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not rows:
        print("no samples")
        return

    # ------------------------------------------------------------ archive
    keys = ["source", "codec", "label", "S1", "S2", "S3", "S4", "qp1", "qp2"]
    for r in rows:
        r["S_fused"] = 0.25 * (r["S1"] + r["S2"] + r["S3"] + r["S4"])
        q1, q2 = r["qp1"], r["qp2"]
        if q1 is None or q2 is None or math.isnan(q1) or math.isnan(q2):
            r["branch"] = "unknown"
        elif q2 > q1:
            r["branch"] = "q2>q1"
        elif q2 == q1:
            r["branch"] = "q2==q1"
        else:
            r["branch"] = "q2<q1"
    keys += ["S_fused", "branch"]
    csv_path = os.path.join(DATA, "pixel_domain_rerun.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    y = np.array([r["label"] for r in rows])
    srcs = np.array([r["source"] for r in rows])
    out = []
    rec = out.append
    rec("pixel-domain rerun with archived per-sample scores")
    rec("  seed %d | %d sources | %d frames @3fps/10s | %dx%d | CRF1=%d dCRF=%d"
        % (SEED, len(picks), N_FRAMES, W, H, CRF1, DCRF))
    rec("  n = %d  (H0 = %d single, H1 = %d double across %s)"
        % (y.size, int(np.sum(y == 0)), int(np.sum(y == 1)), ", ".join(CODECS)))
    rec("  per-sample vector: %s" % csv_path)
    rec("")
    rec("    score        AUC      SE      z       p       95%% CI(HM)   95%% CI(cluster-boot)")
    rec("    " + "-" * 86)
    brng = np.random.default_rng(SEED + 1)
    store = {}
    for name in ("S1", "S2", "S3", "S4", "S_fused"):
        s = np.array([r[name] for r in rows], dtype=np.float64)
        A, se, z, p = auc_mw(y, s)
        lo, hi = cluster_bootstrap_auc(y, s, srcs, brng)
        store[name] = s
        rec("    %-11s %6.4f %7.4f %+7.2f %8.4f   [%.3f, %.3f]   [%.3f, %.3f]"
            % (name, A, se, z, p, A - 1.96 * se, A + 1.96 * se, lo, hi))

    # ---- intra-cluster correlation, now estimable ------------------------
    rec("")
    rec("intra-cluster correlation (one-way ANOVA on within-group residuals):")
    for name in ("S1", "S2", "S3", "S4", "S_fused"):
        s = store[name]
        resid = np.zeros_like(s)
        for c in np.unique(srcs):
            m = srcs == c
            resid[m] = s[m] - s[m].mean()
        rec("    %-11s rho = %s" % (name, "%.4f" % icc_rho(resid, srcs)
                                    if not math.isnan(icc_rho(resid, srcs))
                                    else "n/a"))

    # ---- logistic regression coefficients --------------------------------
    X = np.stack([store[k] for k in ("S1", "S2", "S3", "S4", "S_fused")], axis=1)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    w = logistic_irls(X, y.astype(np.float64))
    rec("")
    rec("logistic regression on the five scores (standardized inputs, IRLS):")
    rec("    intercept      %+0.4f" % w[0])
    for i, nm in enumerate(("S1", "S2", "S3", "S4", "S_fused")):
        rec("    %-13s %+0.4f" % (nm, w[i + 1]))
    rec("    |largest coefficient| / |intercept| = %.3f"
        % (np.max(np.abs(w[1:])) / abs(w[0]) if w[0] != 0 else float("nan")))
    rec("    negative coefficients: %d of %d"
        % (int(np.sum(w[1:] < 0)), len(w) - 1))

    # ---- branch labels ---------------------------------------------------
    bl = [r["branch"] for r in rows if r["label"] == 1]
    rec("")
    rec("branch labels of the %d double-compressed samples (from QP traces):" % len(bl))
    for b in ("q2>q1", "q2==q1", "q2<q1", "unknown"):
        rec("    %-8s %4d  (%.3f)" % (b, bl.count(b), bl.count(b) / max(len(bl), 1)))

    txt = "\n".join(out)
    with open(os.path.join(DATA, "pixel_domain_rerun.txt"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
