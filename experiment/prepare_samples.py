#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import argparse
import subprocess
import time
import shutil

# --- paths ---
# defaults are repo-relative; no host absolute paths
# override with --video-dir / --data-dir / --ffmpeg or env vars
# GCRR_VIDEO_DIR / GCRR_DATA_DIR / GCRR_FFMPEG。
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.environ.get('GCRR_VIDEO_DIR',
                           os.path.join(REPO_ROOT, 'data', 'video'))
DATA_DIR = os.environ.get('GCRR_DATA_DIR',
                          os.path.join(REPO_ROOT, 'data_controlled'))
SINGLE_DIR = os.path.join(DATA_DIR, 'single_compressed')
DOUBLE_BASE = os.path.join(DATA_DIR, 'double_compressed')
INDEX_CSV = os.path.join(DATA_DIR, 'sample_index.csv')

def _default_ffmpeg():
    """Prefer ffmpeg on PATH, then the imageio_ffmpeg binary, else 'ffmpeg'."""
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


FFMPEG = os.environ.get('GCRR_FFMPEG', _default_ffmpeg())

CLIP_DURATION = 30  # seconds trimmed per sample
SEED = 42

# first-pass quality q1 per codec (single arm), fixed
Q1 = {'h264': 28, 'h265': 30, 'vp9': 32}

# q2 offsets around q1, near and far parameter
DEFAULT_Q2_OFFSETS = [-3, 0, 3, 8]

# codec -> (ffmpeg encoder, extra args)
CODEC_CONFIGS = {
    'h264': ('libx264', ['-preset', 'fast', '-pix_fmt', 'yuv420p']),
    'h265': ('libx265', ['-preset', 'fast', '-pix_fmt', 'yuv420p',
                          '-x265-params', 'log-level=error']),
    'vp9':  ('libvpx-vp9', ['-pix_fmt', 'yuv420p', '-deadline', 'realtime',
                              '-cpu-used', '8']),
}


def compress(src, dst, encoder, crf, extra_args, ffmpeg=FFMPEG, duration=CLIP_DURATION):
    """Re-encode the first N seconds with the given codec and CRF."""
    cmd = [
        ffmpeg, '-y', '-i', src,
        '-t', str(duration),
        '-c:v', encoder,
        '-crf', str(crf),
    ] + extra_args + [
        '-an',
        '-movflags', '+faststart',
        dst
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Controlled two-arm sample preparation')
    parser.add_argument('--max', type=int, default=50, help='max number of source videos (default 50)')
    parser.add_argument('--dry-run', action='store_true', help='print the plan without encoding')
    parser.add_argument('--q2-offsets', type=str, default=None,
                        help='comma-separated q2 offsets, e.g. "-3,0,3,8"')
    # paths and hyperparameters
    parser.add_argument('--video-dir', type=str, default=VIDEO_DIR,
                        help=f'source video directory (default {VIDEO_DIR})')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help=f'output root, holds single_compressed/ '
                             f'double_compressed/ sample_index.csv (default {DATA_DIR})')
    parser.add_argument('--ffmpeg', type=str, default=FFMPEG,
                        help=f'ffmpeg executable (default {FFMPEG})')
    parser.add_argument('--duration', type=int, default=CLIP_DURATION,
                        help=f'seconds trimmed per sample (default {CLIP_DURATION})')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'seed for source selection (default {SEED})')
    args = parser.parse_args()

    # derived paths
    video_dir = args.video_dir
    data_dir = args.data_dir
    single_dir = os.path.join(data_dir, 'single_compressed')
    double_base = os.path.join(data_dir, 'double_compressed')
    index_csv = os.path.join(data_dir, 'sample_index.csv')
    clip_duration = args.duration

    # resolve ffmpeg: fall back to PATH when the default is absent
    ffmpeg = args.ffmpeg
    if not os.path.exists(ffmpeg):
        alt = shutil.which('ffmpeg')
        if alt:
            print(f"[ffmpeg] default path missing, using PATH: {alt}")
            ffmpeg = alt
        else:
            print(f"[ffmpeg] warning: default path missing and no ffmpeg on PATH: {ffmpeg}")

    q2_offsets = DEFAULT_Q2_OFFSETS
    if args.q2_offsets:
        q2_offsets = [int(x) for x in args.q2_offsets.split(',') if x.strip() != '']

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    all_videos = sorted(f for f in os.listdir(video_dir) if f.lower().endswith('.mp4'))
    print(f"{len(all_videos)} videos found in video/")

    rng = random.Random(args.seed)
    n_select = min(args.max, len(all_videos))
    selected = rng.sample(all_videos, n_select)
    print(f"selecting {n_select} videos (seed={args.seed})")
    print(f"q2 offsets: {q2_offsets}  (delta_qp = |q2-q1|)")
    print(f"per video per codec: 1 single + {len(q2_offsets)} double")

    plan_total = n_select * len(CODEC_CONFIGS) * (1 + len(q2_offsets))
    print(f"planned samples: {plan_total}")

    if args.dry_run:
        print("\nfirst 5 selected videos:")
        for v in selected[:5]:
            print(f"  {v}")
        return

    os.makedirs(single_dir, exist_ok=True)
    for codec_name in CODEC_CONFIGS:
        os.makedirs(os.path.join(double_base, codec_name), exist_ok=True)
        os.makedirs(os.path.join(single_dir, codec_name), exist_ok=True)

    records = []
    t_start = time.time()
    ok_count = 0
    fail_count = 0

    for i, video_name in enumerate(selected):
        src_path = os.path.join(video_dir, video_name)
        base_name = os.path.splitext(video_name)[0]
        if not os.path.exists(src_path):
            print(f"  [{i+1}/{n_select}] {video_name} - source missing, skipped")
            fail_count += 1
            continue

        sys.stdout.write(f"\r  [{i+1}/{n_select}] {video_name} ")
        sys.stdout.flush()

        for codec_name, (encoder, extra_args) in CODEC_CONFIGS.items():
            q1 = Q1[codec_name]

            # single arm: compress(source, q1)
            single_path = os.path.join(single_dir, codec_name,
                                      f"{base_name}_q1{q1}.mp4")
            if not os.path.exists(single_path):
                try:
                    if compress(src_path, single_path, encoder, q1, extra_args,
                                ffmpeg=ffmpeg, duration=clip_duration):
                        records.append({
                            'video_file': video_name, 'label': 'single',
                            'codec': codec_name, 'q1': str(q1), 'q2': str(q1),
                            'delta_qp': '0',
                            'sample_path': single_path, 'src_path': src_path,
                        })
                        ok_count += 1
                    else:
                        print(f"\n    {codec_name} SINGLE_ENCODE_FAIL")
                        fail_count += 1
                        continue
                except subprocess.TimeoutExpired:
                    print(f"\n    {codec_name} SINGLE_TIMEOUT")
                    fail_count += 1
                    continue
                except Exception as e:
                    print(f"\n    {codec_name} SINGLE_ERR: {e}")
                    fail_count += 1
                    continue
            else:
                records.append({
                    'video_file': video_name, 'label': 'single',
                    'codec': codec_name, 'q1': str(q1), 'q2': str(q1),
                    'delta_qp': '0',
                    'sample_path': single_path, 'src_path': src_path,
                })
                ok_count += 1

            # double arm: compress(single, q2), q2 = q1 + offset
            for off in q2_offsets:
                q2 = q1 + off
                delta = abs(off)
                double_path = os.path.join(
                    double_base, codec_name,
                    f"{base_name}_q1{q1}_q2{q2}.mp4")
                if not os.path.exists(double_path):
                    try:
                        if compress(single_path, double_path, encoder, q2, extra_args,
                                    ffmpeg=ffmpeg, duration=clip_duration):
                            records.append({
                                'video_file': video_name, 'label': 'double',
                                'codec': codec_name, 'q1': str(q1), 'q2': str(q2),
                                'delta_qp': str(delta),
                                'sample_path': double_path, 'src_path': src_path,
                            })
                            ok_count += 1
                        else:
                            print(f"\n    {codec_name} DOUBLE_ENCODE_FAIL(q2={q2})")
                            fail_count += 1
                    except subprocess.TimeoutExpired:
                        print(f"\n    {codec_name} DOUBLE_TIMEOUT(q2={q2})")
                        fail_count += 1
                    except Exception as e:
                        print(f"\n    {codec_name} DOUBLE_ERR(q2={q2}): {e}")
                        fail_count += 1
                else:
                    records.append({
                        'video_file': video_name, 'label': 'double',
                        'codec': codec_name, 'q1': str(q1), 'q2': str(q2),
                        'delta_qp': str(delta),
                        'sample_path': double_path, 'src_path': src_path,
                    })
                    ok_count += 1

    elapsed = time.time() - t_start

    with open(index_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['video_file', 'label', 'codec', 'q1', 'q2',
                           'delta_qp', 'sample_path', 'src_path'])
        writer.writeheader()
        writer.writerows(records)

    n_single = sum(1 for r in records if r['label'] == 'single')
    n_double = sum(1 for r in records if r['label'] == 'double')
    print(f"\n\n{'='*60}")
    print(f"done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  ok: {ok_count}")
    print(f"  failed: {fail_count}")
    print(f"  single: {n_single}")
    print(f"  double: {n_double}")
    for codec_name in CODEC_CONFIGS:
        n = sum(1 for r in records if r['label'] == 'double' and r['codec'] == codec_name)
        print(f"    {codec_name}: {n} double")
    print(f"  index: {index_csv}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

"""Build the controlled two-arm corpus: single = compress(src, q1), double = compress(compress(src, q1), q2)."""

