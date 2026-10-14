#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
120-clip 同语料对照：把 bitstream XGB 放在与像素基线【完全相同规模/选择】的
120-clip 受控子集上，用与主表一致的 source-aware 协议评估。

子集定义（显式、可复现，对应论文"one double variant per source-codec pair"）：
  - 全部 single 片段（每 (video_file, codec) 1 个）-> 60
  - delta_qp == 0 的 double 片段（同参数二次压缩，每 (video_file, codec) 1 个）-> 60
  - 合计 120，1:1 平衡，与像素基线语料对齐。

这样主表"bitstream 0.665 vs 像素 0.566 (+0.099)"在【同语料大小】下可比，
堵住 review 指出的"跨语料对比"口子。

用法：
  python evaluate_subset_120.py
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLIND_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from eval_controlled import load_data, source_aware_cv  # reuse exact protocol

DEFAULT_FEATURES = os.path.join(BLIND_DIR, 'results', 'bitstream_features_controlled.csv')


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEATURES
    # which double variant the baseline used: --max 120 on index ordered
    # (single, then doubles by q2 variant [q1-3, q1, q1+3, q1+8])
    # -> first 120 rows = single + the q1-3 double => delta_qp == 3.
    target_delta = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    X, y, feat_cols, codec_arr, src_arr, delta_arr = load_data(csv_path)

    keep = []
    for i in range(len(y)):
        if y[i] == 0:                      # single -> all
            keep.append(i)
        else:                             # double -> only the matched variant
            if abs(float(delta_arr[i]) - target_delta) < 1e-9:
                keep.append(i)

    Xs = X[keep]
    ys = y[keep]
    srcs = src_arr[keep]
    print(f"120-clip subset: n={len(ys)}  "
          f"single={int((ys == 0).sum())}  double={int((ys == 1).sum())}")
    print(f"  (selected from full 300-clip corpus; "
          f"single=all, double=delta_qp==0)")
    print("=" * 70)
    source_aware_cv(Xs, ys, srcs, feat_cols, model='xgb', ablate=False)


if __name__ == '__main__':
    main()
