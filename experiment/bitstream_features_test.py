import av
import os
import csv
import sys
import numpy as np
from collections import defaultdict, Counter
import time
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLIND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BLIND_DIR)

INDEX_CSV = os.path.join(BLIND_DIR, 'data', 'sample_index_full.csv')
OUTPUT_CSV = os.path.join(BLIND_DIR, 'results', 'bitstream_features_full.csv')


def extract_bitstream_features(video_path, max_frames=300):
    """Extract all bitstream-level features from a video."""
    features = {}
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        ctx = stream.codec_context

        # --- Codec context features ---
        features['codec_name'] = ctx.name
        features['profile'] = str(ctx.profile) if ctx.profile else 'unknown'
        features['level'] = ctx.level if hasattr(ctx, 'level') else 0
        features['width'] = ctx.width
        features['height'] = ctx.height
        features['pixel_format'] = str(ctx.pix_fmt)
        features['bit_rate'] = ctx.bit_rate or 0
        features['duration_s'] = float(stream.duration * stream.time_base) if stream.duration else 0

        # --- Packet-level features (fast, no decode) ---
        packet_sizes = []
        packet_key_flags = []
        packet_pts = []
        
        # Reset container
        container.seek(0)
        
        for packet in container.demux(stream):
            if packet.size > 0:
                packet_sizes.append(packet.size)
                packet_key_flags.append(1 if packet.is_keyframe else 0)
                if packet.pts is not None:
                    packet_pts.append(packet.pts)

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
            # Entropy of packet size distribution
            probs = ps / ps.sum()
            probs = probs[probs > 0]
            features['pkt_size_entropy'] = float(-np.sum(probs * np.log2(probs)))
            # Percentiles
            for p in [10, 25, 75, 90]:
                features[f'pkt_size_p{p}'] = float(np.percentile(ps, p))

        if packet_key_flags:
            features['key_frame_count'] = sum(packet_key_flags)
            features['key_frame_ratio'] = sum(packet_key_flags) / len(packet_key_flags)
        
        # GOP structure: distance between consecutive key frames
        if packet_key_flags:
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

        # PTS delta statistics
        if packet_pts:
            pts_deltas = np.diff(packet_pts)
            if len(pts_deltas) > 0:
                features['pts_delta_mean'] = float(np.mean(pts_deltas))
                features['pts_delta_std'] = float(np.std(pts_deltas))

        # --- Frame-level features (decode required) ---
        container2 = av.open(video_path)
        stream2 = container2.streams.video[0]
        
        frame_types = []  # pict_type: 1=I, 2=P, 3=B, 0=?
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
            # I=1, P=2, B=3
            features['i_ratio'] = type_counter.get(1, 0) / total
            features['p_ratio'] = type_counter.get(2, 0) / total
            features['b_ratio'] = type_counter.get(3, 0) / total
            features['other_ratio'] = (total - type_counter.get(1, 0) - type_counter.get(2, 0) - type_counter.get(3, 0)) / total
            features['frame_type_entropy'] = float(_entropy(list(type_counter.values())))
            # I-frame spacing
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

        # --- Key frame size vs non-key frame size ---
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

        # --- Derived composite features ---
        if features.get('pkt_size_mean', 0) > 0 and features.get('n_packets', 0) > 0:
            features['bits_per_pixel'] = features['pkt_size_mean'] * 8 * features['n_packets'] / (
                features['width'] * features['height'] * max(features['n_frames_decoded'], 1))
        
        if features.get('bit_rate', 0) > 0 and features.get('duration_s', 0) > 0:
            features['actual_bitrate'] = features['bit_rate']
        
        features['error'] = ''

    except Exception as e:
        features['error'] = str(e)
        # Set defaults
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
    # Load sample index
    meta = list(csv.DictReader(open(INDEX_CSV, 'r', encoding='utf-8')))
    print(f"Total entries: {len(meta)}")

    # Filter existing files
    valid = [(row, row['sample_path']) for row in meta 
             if row.get('sample_path') and os.path.exists(row['sample_path'])]
    print(f"Valid files: {len(valid)}")

    # Numeric feature keys
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
        feats['video_file'] = row.get('video_file', '')
        feats['label'] = row.get('label', '')
        feats['codec'] = row.get('codec', '')
        feats['crf'] = row.get('crf', '')
        feats['sample_path'] = vpath
        all_features.append(feats)

        # Incremental save every 20
        if len(all_features) % 20 == 0:
            _save_csv(all_features, numeric_keys)

    # Final save
    _save_csv(all_features, numeric_keys)

    # === Analysis ===
    print(f"\n{'='*70}")
    print("BITSTREAM FEATURE ANALYSIS: SINGLE vs DOUBLE")
    print(f"{'='*70}")

    singles = [f for f in all_features if f['label'] == 'single']
    doubles = [f for f in all_features if f['label'] == 'double']
    print(f"SINGLE: {len(singles)}, DOUBLE: {len(doubles)}")

    # Per-codec breakdown
    by_codec = defaultdict(list)
    for f in all_features:
        by_codec[f['codec']].append(f)
    for codec, items in sorted(by_codec.items()):
        n_s = sum(1 for x in items if x['label'] == 'single')
        n_d = sum(1 for x in items if x['label'] == 'double')
        print(f"  {codec}: single={n_s}, double={n_d}")

    # Feature comparison
    print(f"\n{'Feature':>22s} {'SINGLE':>10s} {'DOUBLE':>10s} {'Ratio':>8s} {'Separability':>12s}")
    print("-" * 65)

    for k in numeric_keys:
        sv = [f.get(k, 0) for f in singles if isinstance(f.get(k, 0), (int, float)) and f.get(k, 0) != 0]
        dv = [f.get(k, 0) for f in doubles if isinstance(f.get(k, 0), (int, float)) and f.get(k, 0) != 0]
        sm = np.mean(sv) if sv else 0
        dm = np.mean(dv) if dv else 0
        
        # Separability: Cohen's d
        if sv and dv and len(sv) > 1 and len(dv) > 1:
            pooled_std = np.sqrt((np.var(sv) + np.var(dv)) / 2)
            cohen_d = abs(np.mean(sv) - np.mean(dv)) / (pooled_std + 1e-10)
            sep_str = f"d={cohen_d:.3f}"
        else:
            cohen_d = 0
            sep_str = "N/A"
        
        ratio = sm / (dm + 1e-10) if dm != 0 else 0
        print(f"{k:>22s} {sm:10.4f} {dm:10.4f} {ratio:8.3f} {sep_str:>12s}")

    # AUC computation per feature
    print(f"\n{'='*70}")
    print("PER-FEATURE AUC (Mann-Whitney U)")
    print(f"{'='*70}")
    print(f"{'Feature':>22s} {'AUC':>8s} {'|AUC-0.5|':>10s}")
    print("-" * 42)

    auc_results = {}
    for k in numeric_keys:
        sv = np.array([f.get(k, 0) for f in singles if isinstance(f.get(k, 0), (int, float))])
        dv = np.array([f.get(k, 0) for f in doubles if isinstance(f.get(k, 0), (int, float))])
        if len(sv) > 0 and len(dv) > 0:
            auc = _fast_auc(sv, dv)
            auc_results[k] = auc
            gap = abs(auc - 0.5)
            marker = " ***" if gap > 0.1 else " **" if gap > 0.05 else ""
            print(f"{k:>22s} {auc:8.4f} {gap:10.4f}{marker}")

    # Top features
    sorted_auc = sorted(auc_results.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
    print(f"\n{'='*70}")
    print("TOP 10 FEATURES BY |AUC - 0.5|")
    print(f"{'='*70}")
    for k, auc in sorted_auc[:10]:
        print(f"  {k:>22s}: AUC={auc:.4f} (|d|={abs(auc-0.5):.4f})")

    # Cross-codec AUC
    print(f"\n{'='*70}")
    print("CROSS-CODEC AUC BREAKDOWN")
    print(f"{'='*70}")
    for codec in ['h264', 'h265', 'vp9']:
        print(f"\n--- {codec} double vs single ---")
        s_items = [f for f in singles]
        d_items = [f for f in doubles if f['codec'] == codec]
        if not d_items:
            print(f"  No {codec} double samples")
            continue
        
        print(f"  Single: {len(s_items)}, {codec} double: {len(d_items)}")
        codec_aucs = []
        for k in numeric_keys[:15]:
            sv = np.array([f.get(k, 0) for f in s_items])
            dv = np.array([f.get(k, 0) for f in d_items])
            if len(sv) > 0 and len(dv) > 0:
                auc = _fast_auc(sv, dv)
                codec_aucs.append((k, auc))
        
        codec_aucs.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        for k, auc in codec_aucs[:5]:
            print(f"    {k:>22s}: AUC={auc:.4f}")


def _fast_auc(y0, y1):
    """Fast AUC via Mann-Whitney U."""
    n0, n1 = len(y0), len(y1)
    if n0 == 0 or n1 == 0:
        return 0.5
    all_vals = np.concatenate([y0, y1])
    ranks = np.argsort(np.argsort(all_vals)).astype(float)
    # Handle ties
    unique, inverse = np.unique(all_vals, return_inverse=True)
    for i, u in enumerate(unique):
        mask = inverse == i
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r0 = ranks[:n0].sum()
    u_stat = r0 - n0 * (n0 + 1) / 2
    auc = 1 - u_stat / (n0 * n1)
    return auc


def _save_csv(features, numeric_keys):
    all_keys = sorted(set().union(*(f.keys() for f in features)))
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        for feat in features:
            writer.writerow(feat)


if __name__ == '__main__':
    main()

"""Extract bitstream-level features that are invisible in the pixel domain."""

