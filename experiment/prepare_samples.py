#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盲检测受控双臂样本生成 (controlled two-arm generation)
========================================================

为什么改：旧版 single = 对源视频做 stream-copy（不重编码），其真实编码格式
未知（codec='original'），而 double = 用 libx264/265/vp9 在固定 CRF 重编码。
于是 "single vs double" 对比里分类器可能偷学【编解码器身份】，而不是双压缩
痕迹 —— 这是方法成立的根基性漏洞（codec-ID 泄漏 + 出处失控）。

新设计（标准且最硬的做法）：两个臂都用【自己的编码器、已知参数】生成
  - 负样本 single : compress(source, q1)                  —— 单次压缩, 首压参数 q1 已知
  - 正样本 double : compress(compress(source,q1), q2)     —— 在 single 基础上二次压缩, q2 已知
两个臂都在【同一码器、同一 q1】下生成，因此 single-vs-double 是纯净的
"双量化" 测试，不存在 codec-ID 泄漏，也不依赖源视频的未知出处。

近参数扫描：q2 围绕 q1 取值（含 q2==q1 的极端近参数 delta=0），用于
刻画可检测性边界（配合 eval_controlled.py --max-delta 即可做近参数研究）。

输出：
  - blind_detection/data_controlled/single_compressed/{codec}/   负样本
  - blind_detection/data_controlled/double_compressed/{codec}/  正样本
  - blind_detection/data_controlled/sample_index.csv
        列: video_file, label, codec, q1, q2, delta_qp, sample_path, src_path

用法：
    py prepare_samples.py                 # 默认 50 个视频
    py prepare_samples.py --max 10        # 小规模试跑
    py prepare_samples.py --dry-run       # 只打印计划, 不编码
    py prepare_samples.py --q2-offsets "-2,0,3,8"
    # 换机器/换数据根目录时, 用命令行参数覆盖写死路径 (提升可复现性):
    py prepare_samples.py \
        --video-dir  /path/to/source_videos \
        --data-dir   /path/to/blind_detection/data_controlled \
        --ffmpeg     /usr/bin/ffmpeg \
        --duration   30 \
        --seed       42
    # 若 --ffmpeg 未给且默认镜像路径不存在, 会自动回退到 PATH 上的 ffmpeg
"""

import os
import sys
import csv
import random
import argparse
import subprocess
import time
import shutil

# ── 路径配置 ──────────────────────────────────────────────
# 默认值一律相对仓库根：仓库里不含任何本机绝对路径。
# 换机只需传参 --video-dir / --data-dir / --ffmpeg，或设环境变量
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
    """PATH 上的 ffmpeg 优先；其次 imageio_ffmpeg 自带的二进制；都没有就留 'ffmpeg'。"""
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


FFMPEG = os.environ.get('GCRR_FFMPEG', _default_ffmpeg())

CLIP_DURATION = 30  # 每个样本取前30秒
SEED = 42

# 每个码器的【首压质量 q1】（single 臂用）—— 固定、已知
Q1 = {'h264': 28, 'h265': 30, 'vp9': 32}

# 二次压缩质量 q2 相对 q1 的偏移扫描（含近参数与远参数）
# delta_qp = |q2 - q1|；偏移 0 即「同质量二次压缩」，是极端近参数情形
DEFAULT_Q2_OFFSETS = [-3, 0, 3, 8]

# 码器 -> (ffmpeg 编码器, 额外参数)
CODEC_CONFIGS = {
    'h264': ('libx264', ['-preset', 'fast', '-pix_fmt', 'yuv420p']),
    'h265': ('libx265', ['-preset', 'fast', '-pix_fmt', 'yuv420p',
                          '-x265-params', 'log-level=error']),
    'vp9':  ('libvpx-vp9', ['-pix_fmt', 'yuv420p', '-deadline', 'realtime',
                              '-cpu-used', '8']),
}


def compress(src, dst, encoder, crf, extra_args, ffmpeg=FFMPEG, duration=CLIP_DURATION):
    """用 ffmpeg 以指定码器/CRF 重编码截取前 N 秒（产生一次压缩）。"""
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
    parser = argparse.ArgumentParser(description='Blind-GCRR controlled two-arm sample preparation')
    parser.add_argument('--max', type=int, default=50, help='最大视频数 (默认50)')
    parser.add_argument('--dry-run', action='store_true', help='只打印计划不执行')
    parser.add_argument('--q2-offsets', type=str, default=None,
                        help='q2 偏移列表, 逗号分隔, 如 "-3,0,3,8"')
    # ── 路径/超参参数化 (默认沿用写死值, 换机只需传参) ──
    parser.add_argument('--video-dir', type=str, default=VIDEO_DIR,
                        help=f'源视频目录 (默认 {VIDEO_DIR})')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help=f'输出数据根目录, 内含 single_compressed/ '
                             f'double_compressed/ sample_index.csv (默认 {DATA_DIR})')
    parser.add_argument('--ffmpeg', type=str, default=FFMPEG,
                        help=f'ffmpeg 可执行文件路径 (默认 {FFMPEG})')
    parser.add_argument('--duration', type=int, default=CLIP_DURATION,
                        help=f'每个样本截取秒数 (默认 {CLIP_DURATION})')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'随机选取源视频的种子 (默认 {SEED})')
    args = parser.parse_args()

    # ── 运行时解析派生路径 ──
    video_dir = args.video_dir
    data_dir = args.data_dir
    single_dir = os.path.join(data_dir, 'single_compressed')
    double_base = os.path.join(data_dir, 'double_compressed')
    index_csv = os.path.join(data_dir, 'sample_index.csv')
    clip_duration = args.duration

    # ffmpeg 解析: 若默认镜像路径不存在, 回退到 PATH 上的 ffmpeg
    ffmpeg = args.ffmpeg
    if not os.path.exists(ffmpeg):
        alt = shutil.which('ffmpeg')
        if alt:
            print(f"[ffmpeg] 默认路径不存在, 回退到 PATH: {alt}")
            ffmpeg = alt
        else:
            print(f"[ffmpeg] 警告: 默认路径不存在且 PATH 中无 ffmpeg: {ffmpeg}")

    q2_offsets = DEFAULT_Q2_OFFSETS
    if args.q2_offsets:
        q2_offsets = [int(x) for x in args.q2_offsets.split(',') if x.strip() != '']

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    all_videos = sorted(f for f in os.listdir(video_dir) if f.lower().endswith('.mp4'))
    print(f"video/ 目录共 {len(all_videos)} 个视频")

    rng = random.Random(args.seed)
    n_select = min(args.max, len(all_videos))
    selected = rng.sample(all_videos, n_select)
    print(f"随机选取 {n_select} 个视频 (seed={args.seed})")
    print(f"q2 偏移扫描: {q2_offsets}  (delta_qp = |q2-q1|)")
    print(f"每视频每码器生成: 1 single + {len(q2_offsets)} double")

    plan_total = n_select * len(CODEC_CONFIGS) * (1 + len(q2_offsets))
    print(f"预计生成 {plan_total} 个样本")

    if args.dry_run:
        print("\n前 5 个选中视频:")
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
            print(f"  [{i+1}/{n_select}] {video_name} — 源文件不存在，跳过")
            fail_count += 1
            continue

        sys.stdout.write(f"\r  [{i+1}/{n_select}] {video_name} ")
        sys.stdout.flush()

        for codec_name, (encoder, extra_args) in CODEC_CONFIGS.items():
            q1 = Q1[codec_name]

            # ---- single 臂: compress(source, q1) ----
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

            # ---- double 臂: compress(single, q2), q2 = q1 + offset ----
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
    print(f"样本生成完成! 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  成功: {ok_count} 个样本")
    print(f"  失败: {fail_count} 个")
    print(f"  单次压缩(single): {n_single} 个")
    print(f"  双次压缩(double): {n_double} 个")
    for codec_name in CODEC_CONFIGS:
        n = sum(1 for r in records if r['label'] == 'double' and r['codec'] == codec_name)
        print(f"    {codec_name}: {n} double")
    print(f"  索引文件: {index_csv}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
