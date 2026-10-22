#!/usr/bin/env python
"""M1 (major comment 1): calibrate the CRF / QP axis onto Theorem 6's epsilon.

Theorem 6 is proved for a *scalar* pair (q1, q2) of uniform mid-tread
quantization steps.  Experiments, however, vary a rate factor (CRF).  This
script makes the bridge explicit and quantifies how much the bridge distorts
the epsilon that enters the barrier.

Chain
-----
    CRF --(rate control)--> QP_frame --(H.264 standard)--> Qstep --(ratio)--> eps

1. H.264 step-size law (ITU-T H.264, Table 8-... / the Qstep doubling rule):

       Qstep(QP) = Qstep(0) * 2^(QP/6),      Qstep(0) = 0.625,

   so a QP difference of d multiplies the step by 2^(d/6) and therefore

       eps(d) = 2^(d/6) - 1.

2. Frame-type offsets.  x264 converts its ratios into QP offsets through
   qscale = 0.85 * 2^((QP-12)/6) and qscale_I = qscale/ipratio with defaults
   ipratio = 1.40, pbratio = 1.30:

       dQP_I = -6 log2(1.40) = -2.91,   dQP_B = +6 log2(1.30) = +2.28.

3. Spatial / temporal adaptive quantization.  x264 aq-mode 1 and MB-tree both
   perturb the per-block QP; the documented clamp is +-2 (aq-strength 1.0).
   We model the residual offset as uniform on [-U, U] with U = 2.

The *nominal* eps uses only the CRF difference; the *effective* eps is
per-frame, because both passes draw their own frame type and AQ offset.  The
pooled barrier is governed by the root-mean-square of the per-frame eps, since
H^2 ~ C eps^2 per frame and Hellinger-type quantities accumulate additively
over frames.

Outputs a table to stdout and to data/epsilon_calibration.txt.
"""
from __future__ import annotations

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")
SEED = 20261006

QSTEP0 = 0.625           # H.264 Qstep at QP = 0
IPRATIO, PBRATIO = 1.40, 1.30
DQP_I = -6.0 * math.log2(IPRATIO)
DQP_B = +6.0 * math.log2(PBRATIO)
AQ_CLAMP = 2.0           # +-2 QP from AQ / MB-tree (x264 documented clamp)
# Frame-type mix of a typical x264 encode with bframes=3.
PICT_MIX = [("I", 0.02, DQP_I), ("P", 0.33, 0.0), ("B", 0.65, DQP_B)]
N_MC = 400_000


def qstep(qp: float) -> float:
    """H.264 quantization step; doubles every 6 QP."""
    return QSTEP0 * 2.0 ** (qp / 6.0)


def eps_from_dqp(d_qp: float) -> float:
    """eps = q2/q1 - 1 induced by a QP difference d_qp."""
    return 2.0 ** (d_qp / 6.0) - 1.0


def sample_offsets(rng: np.random.Generator, n: int) -> np.ndarray:
    """Per-frame QP offset: frame type + AQ/MB-tree, drawn from the mix."""
    probs = np.array([p for _, p, _ in PICT_MIX])
    probs = probs / probs.sum()
    idx = rng.choice(len(PICT_MIX), size=n, p=probs)
    dqp_type = np.array([PICT_MIX[i][2] for i in idx])
    dqp_aq = rng.uniform(-AQ_CLAMP, AQ_CLAMP, size=n)
    return dqp_type + dqp_aq


def main() -> int:
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lines: list[str] = []
    rec = lines.append

    rec("epsilon calibration: CRF -> QP -> Qstep -> eps   (seed %d, %d MC draws)"
        % (SEED, N_MC))
    rec("")
    rec("H.264 step law      : Qstep(QP) = %.3f * 2^(QP/6)" % QSTEP0)
    rec("x264 frame-type offs: dQP_I = %+.2f  dQP_P = %+.2f  dQP_B = %+.2f"
        % (DQP_I, 0.0, DQP_B))
    rec("AQ / MB-tree clamp  : +-%s QP (uniform model)" % AQ_CLAMP)
    rec("")
    hdr = ("%8s %12s %12s %12s %12s %12s %12s %12s"
           % ("dCRF", "eps_nominal", "eps_mean", "eps_rms", "rms/nom",
              "P(eps<=0)", "N*/N*_nom", "eps_p05..p95"))
    rec(hdr)
    rec("-" * len(hdr))

    for d_crf in (1, 2, 3, 4, 6):
        u1 = sample_offsets(rng, N_MC)
        u2 = sample_offsets(rng, N_MC)
        d_qp_eff = d_crf + (u2 - u1)
        eps_eff = 2.0 ** (d_qp_eff / 6.0) - 1.0
        eps_nom = eps_from_dqp(float(d_crf))
        eps_rms = float(np.sqrt(np.mean(eps_eff ** 2)))
        # pooled sample complexity N* ~ eps^-2 relative to the nominal reading
        ratio = (eps_nom / eps_rms) ** 2 if eps_rms > 0 else float("inf")
        p_neg = float(np.mean(eps_eff <= 0.0))
        lo, hi = np.percentile(eps_eff, [5, 95])
        rec("%8d %12.4f %12.4f %12.4f %12.3f %12.3f %12.3f  %+.3f..%+.3f"
            % (d_crf, eps_nom, float(np.mean(eps_eff)), eps_rms,
               eps_rms / eps_nom, p_neg, ratio, lo, hi))

    rec("")
    rec("Reading:")
    rec("  * eps_nominal is what a CRF-based reading of Theorem 6 would assume.")
    rec("  * eps_rms is what Theorem 6 actually sees after pooling frames, since")
    rec("    H^2 accumulates additively and H^2 ~ C eps^2 per frame.")
    rec("  * N*/N*_nom is the resulting distortion of the barrier: values far")
    rec("    from 1 mean the CRF axis cannot be substituted for the step axis.")
    rec("  * P(eps<=0) is the fraction of frame pairs for which the SECOND step")
    rec("    is not larger than the first, i.e. the q2>q1 regime of Theorem 1")
    rec("    does not hold frame-locally even when dCRF > 0.")

    # --- worked arithmetic for the smallest reported gap -------------------
    rec("")
    rec("Worked example at dCRF = 3 (the empirical study's 'Delta CRF <= 3'):")
    d = 3
    eps_nom = eps_from_dqp(float(d))
    rec("  Qstep ratio from CRF alone      : 2^(3/6) = %.4f -> eps = %.4f"
        % (2 ** 0.5, eps_nom))
    for dqp in (-6, -3, 0, +3, +6):
        e = eps_from_dqp(float(d + dqp))
        rec("    dQP_eff = %+d  ->  q2/q1 = %.4f  ->  eps = %+.4f"
            % (d + dqp, 2 ** ((d + dqp) / 6.0), e))
    rec("  A frame-type/AQ swing of only +-3 QP therefore moves eps by more than")
    rec("  a factor of two at dCRF = 3, and a swing of -6 QP flips its sign.")

    out = "\n".join(lines)
    print(out)
    with open(os.path.join(DATA, "epsilon_calibration.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
