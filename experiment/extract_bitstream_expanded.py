#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitstream feature extraction for the EXPANDED blind detection dataset.

读取: 由 --index 指定 (默认 data_controlled/sample_index.csv)
写出: 由 --output 指定 (默认 results/bitstream_features_controlled.csv)
  注: 默认路径相对本脚本位置解析 (../data_controlled, ../results),
  故在任意机器上从项目根目录直接 `py scripts/extract_bitstream_expanded.py`
  即可一键运行, 无需改路径。

为配合 prepare_samples.py 的【受控双臂】新 index，新增 --index / --output /
--max 参数：脚本会把 index 里的 meta 列 (video_file, label, codec, q1, q2,
delta_qp, crf, sample_path) 原样透传进特征表，供 eval_controlled.py 使用。

两步工作流 (受控双臂):
  py prepare_samples.py --max 20
  py extract_bitstream_expanded.py \
        --index data_controlled/sample_index.csv \
        --output results/bitstream_features_controlled.csv
  py eval_controlled.py --features results/bitstream_features_controlled.csv
"""

import av
import os
import csv
import sys
import argparse
import numpy as np
from collections import defaultdict, Counter
import time

# 路径相对本脚本位置解析 (scripts/ -> 项目根 blind_detection/),
# 换机器无需改路径, 从项目根目录直接运行即可。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLIND_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_INDEX = os.path.join(BLIND_DIR, 'data_controlled', 'sample_index.csv')
DEFAULT_OUTPUT = os.path.join(BLIND_DIR, 'results', 'bitstream_features_controlled.csv')


def extract_bitstream_features(video_path, max_frames=300, max_packets=20000):
    """Extract all bitstream-level features from a video.
    
    Args:
        video_path: path to video file
        max_frames: max frames to decode for frame-type analysis
        max_packets: max packets to demux for packet-level statistics
                   (safety limit for very long videos)
    """
    features = {}
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        ctx = stream.codec_context

        features['codec_name'] = ctx.name
        features['profile'] = str(ctx.profile) if ctx.profile else 'unknown'
        features['level'] = ctx.level if hasattr(ctx, 'level') else 0
        features['width'] = ctx.width
        features['height'] = ctx.height
        features['pixel_format'] = str(ctx.pix_fmt)
        features['bit_rate'] = ctx.bit_rate or 0
        features['duration_s'] = float(stream.duration * stream.time_base) if stream.duration else 0

        packet_sizes = []
        packet_key_flags = []
        packet_pts = []
        
        container.seek(0)
        for packet in container.demux(stream):
            if packet.size > 0:
                packet_sizes.append(packet.size)
                packet_key_flags.append(1 if packet.is_keyframe else 0)
                if packet.pts is not None:
                    packet_pts.append(packet.pts)
            if len(packet_sizes) >= max_packets:
                break

        n_packets = len(packet_sizes)
        features['n_packets'] = n_packets

        if packet_sizes:
            ps = np.array(packet_sizes, dtype=np.float64)
            features['pkt_size_mean'] = float(np.mean(ps))
            features['pkt_size_std'] = float(np.std(ps))
            features['pkt_size_median'] = float(np.median(ps))
            features['pkt_size_max'] = float(np.max(ps))
            features['pkt_size_min'] = float(np.min(ps))
            features['pkt_size_cv'] = float(np.std(ps) / (np.mean(ps) + 1e-10))
            features['pkt_size_skew'] = float(_skewness(ps))
            features['pkt_size_kurtosis'] = float(_kurtosis(ps))
            probs = ps / ps.sum()
            probs = probs[probs > 0]
            features['pkt_size_entropy'] = float(-np.sum(probs * np.log2(probs)))
            for p in [10, 25, 75, 90]:
                features[f'pkt_size_p{p}'] = float(np.percentile(ps, p))

        if packet_key_flags:
            features['key_frame_count'] = sum(packet_key_flags)
            features['key_frame_ratio'] = sum(packet_key_flags) / len(packet_key_flags)
            key_indices = [i for i, k in enumerate(packet_key_flags) if k]
            if len(key_indices) > 1:
                gop_lengths = np.diff(key_indices)
                features['gop_mean'] = float(np.mean(gop_lengths))
                features['gop_std'] = float(np.std(gop_lengths))
                features['gop_min'] = int(np.min(gop_lengths))
                features['gop_max'] = int(np.max(gop_lengths))
                features['gop_cv'] = float(np.std(gop_lengths) / (np.mean(gop_lengths) + 1e-10))
            else:
                features['gop_mean'] = 0
                features['gop_std'] = 0
                features['gop_min'] = 0
                features['gop_max'] = 0
                features['gop_cv'] = 0

        if packet_pts:
            pts_deltas = np.diff(packet_pts)
            if len(pts_deltas) > 0:
                features['pts_delta_mean'] = float(np.mean(pts_deltas))
                features['pts_delta_std'] = float(np.std(pts_deltas))

        container2 = av.open(video_path)
        stream2 = container2.streams.video[0]
        frame_types = []
        frame_count = 0
        for frame in container2.decode(stream2):
            frame_types.append(frame.pict_type)
            frame_count += 1
            if frame_count >= max_frames:
                break
        container2.close()
        
        features['n_frames_decoded'] = frame_count
        
        if frame_types:
            type_counter = Counter(frame_types)
            total = len(frame_types)
            features['i_ratio'] = type_counter.get(1, 0) / total
            features['p_ratio'] = type_counter.get(2, 0) / total
            features['b_ratio'] = type_counter.get(3, 0) / total
            features['other_ratio'] = (total - type_counter.get(1, 0) - type_counter.get(2, 0) - type_counter.get(3, 0)) / total
            features['frame_type_entropy'] = float(_entropy(list(type_counter.values())))
            i_positions = [i for i, t in enumerate(frame_types) if t == 1]
            if len(i_positions) > 1:
                i_spacings = np.diff(i_positions)
                features['i_spacing_mean'] = float(np.mean(i_spacings))
                features['i_spacing_std'] = float(np.std(i_spacings))
                features['i_spacing_cv'] = float(np.std(i_spacings) / (np.mean(i_spacings) + 1e-10))
            else:
                features['i_spacing_mean'] = 0
                features['i_spacing_std'] = 0
                features['i_spacing_cv'] = 0

        if packet_sizes and packet_key_flags:
            key_sizes = [s for s, k in zip(packet_sizes, packet_key_flags) if k]
            nonkey_sizes = [s for s, k in zip(packet_sizes, packet_key_flags) if not k]
            if key_sizes:
                features['key_size_mean'] = float(np.mean(key_sizes))
                features['key_size_std'] = float(np.std(key_sizes))
            if nonkey_sizes:
                features['nonkey_size_mean'] = float(np.mean(nonkey_sizes))
                features['nonkey_size_std'] = float(np.std(nonkey_sizes))
            if key_sizes and nonkey_sizes:
                features['key_nonkey_ratio'] = float(np.mean(key_sizes) / (np.mean(nonkey_sizes) + 1e-10))

        if features.get('pkt_size_mean', 0) > 0 and features.get('n_packets', 0) > 0:
            features['bits_per_pixel'] = features['pkt_size_mean'] * 8 * features['n_packets'] / (
                features['width'] * features['height'] * max(features['n_frames_decoded'], 1))
        
        if features.get('bit_rate', 0) > 0 and features.get('duration_s', 0) > 0:
            features['actual_bitrate'] = features['bit_rate']
        
        features['error'] = ''

    except Exception as e:
        features['error'] = str(e)
        for k in ['n_packets', 'pkt_size_mean', 'pkt_size_std', 'pkt_size_cv',
                   'key_frame_ratio', 'gop_mean', 'gop_cv', 'i_ratio', 'p_ratio',
                   'b_ratio', 'key_nonkey_ratio', 'bits_per_pixel']:
            if k not in features:
                features[k] = 0

    return features


def _skewness(arr):
    if len(arr) < 3:
        return 0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0
    return float(np.mean(((arr - m) / s) ** 3))


def _kurtosis(arr):
    if len(arr) < 4:
        return 0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0
    return float(np.mean(((arr - m) / s) ** 4) - 3)


def _entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0
    probs = [c / total for c in counts if c > 0]
    return float(-sum(p * np.log2(p) for p in probs))


def main():
    ap = argparse.ArgumentParser(
        description='Bitstream feature extraction (supports controlled two-arm index)')
    ap.add_argument('--index', type=str, default=DEFAULT_INDEX,
                    help='sample_index.csv 路径 (默认 data_expanded/sample_index.csv). '
                         '受控双臂请用 data_controlled/sample_index.csv')
    ap.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                    help='输出特征 CSV 路径 (默认 bitstream_features_expanded.csv). '
                         '受控双臂建议 results/bitstream_features_controlled.csv')
    ap.add_argument('--max', type=int, default=None,
                    help='最多处理 N 个样本 (试跑/调试用)')
    args = ap.parse_args()

    meta = list(csv.DictReader(open(args.index, 'r', encoding='utf-8')))
    print(f"Index : {args.index}")
    print(f"Total entries: {len(meta)}")

    valid = [(row, row['sample_path']) for row in meta 
             if row.get('sample_path') and os.path.exists(row['sample_path'])]
    print(f"Valid files: {len(valid)}")
    if args.max is not None:
        valid = valid[:args.max]
        print(f"--max 限制: 处理前 {len(valid)} 个")

    numeric_keys = [
        'n_packets', 'pkt_size_mean', 'pkt_size_std', 'pkt_size_median',
        'pkt_size_max', 'pkt_size_min', 'pkt_size_cv', 'pkt_size_skew',
        'pkt_size_kurtosis', 'pkt_size_entropy',
        'pkt_size_p10', 'pkt_size_p25', 'pkt_size_p75', 'pkt_size_p90',
        'key_frame_count', 'key_frame_ratio',
        'gop_mean', 'gop_std', 'gop_min', 'gop_max', 'gop_cv',
        'pts_delta_mean', 'pts_delta_std',
        'n_frames_decoded', 'i_ratio', 'p_ratio', 'b_ratio', 'other_ratio',
        'frame_type_entropy', 'i_spacing_mean', 'i_spacing_std', 'i_spacing_cv',
        'key_size_mean', 'key_size_std', 'nonkey_size_mean', 'nonkey_size_std',
        'key_nonkey_ratio', 'bits_per_pixel',
        'bit_rate', 'width', 'height', 'duration_s'
    ]

    all_features = []
    t_start = time.time()

    for i, (row, vpath) in enumerate(valid):
        elapsed = time.time() - t_start
        fps = (i + 1) / (elapsed + 1e-10)
        eta = (len(valid) - i - 1) / fps if fps > 0 else 0
        print(f"[{i+1}/{len(valid)}] {os.path.basename(vpath):15s} "
              f"label={row.get('label','?'):6s} codec={row.get('codec','?'):8s} "
              f"ETA={eta:.0f}s", flush=True)

        feats = extract_bitstream_features(vpath)
        # ── meta 列原样透传 (供 eval_controlled.py 使用) ──
        feats['video_file'] = row.get('video_file', '')
        feats['label'] = row.get('label', '')
        feats['codec'] = row.get('codec', '')
        feats['q1'] = row.get('q1', '')               # 受控双臂: 首压参数
        feats['q2'] = row.get('q2', '')               # 受控双臂: 二次压参数
        feats['delta_qp'] = row.get('delta_qp', '')   # |q2 - q1|
        feats['crf'] = row.get('crf', '')             # 旧 expanded index 兼容
        feats['sample_path'] = vpath
        all_features.append(feats)

        if (i + 1) % 20 == 0:
            _save_csv(all_features, numeric_keys, args.output)

    _save_csv(all_features, numeric_keys, args.output)
    print(f"\nSaved to: {args.output}")


def _save_csv(features, numeric_keys, out_csv):
    all_keys = sorted(set().union(*(f.keys() for f in features)))
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        for feat in features:
            writer.writerow(feat)


if __name__ == '__main__':
    main()
