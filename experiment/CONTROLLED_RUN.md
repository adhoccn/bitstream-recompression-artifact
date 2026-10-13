# 受控双臂三步 + Su2018 受控重跑 —— 本机运行手册

> 沙箱环境无 `ffmpeg` / `PyAV`，真实视频编码与像素解码只能在你本机跑。
> 本文件给出一条命令串：先生成受控双臂数据，再抽码流域特征、做统一评估，
> 最后用受控数据重跑 Wang & Farid (2006) 像素基线，把旧稿的 `0.503` 换成受控值。
>
> 前提：本机已装 `ffmpeg`(在 PATH) + `PyAV` + `opencv-python`(`cv2`，步骤 4 需要) + `scikit-learn` + `xgboost`。

> ⚠ **路径说明**：下面是**原始实验现场**的记录，工作目录是
> `C:/baidunetdiskdownload/lightweight/blind_detection`，脚本在 `scripts/` 下。
> 在本仓库里，同样的脚本位于 `experiment/`，工作目录是仓库根，
> 命令要相应改成 `py experiment/prepare_samples.py ...`（见 `REPRODUCE.md`）。
> 保留原路径是为了让这份现场记录可对照，不要照抄执行。

```bash
# 0. 进入项目 (在你本机, 不是沙箱)
cd C:/baidunetdiskdownload/lightweight/blind_detection

# 1. 生成受控双臂数据
#    single = compress(src, q1)
#    double = compress(compress(src, q1), q2),  q2 围绕 q1 偏移扫描
#    先试小批量确认管线无误:
python scripts/prepare_samples.py --max 20
#    全量: 受控语料实际用 --max 20 生成
#    (20 源 × [1 single + 4 double] × 3 码器 = 300 clip):
# python scripts/prepare_samples.py --max 20

# 2. 抽码流域特征 (指向受控 index)
python scripts/extract_bitstream_expanded.py \
    --index data_controlled/sample_index.csv \
    --output results/bitstream_features_controlled.csv

# 3. 码流域统一评估 (source-aware CV + 同码器双臂跨码器 + 码率消融 + EER/TDR@FPR + 95%CI)
python scripts/eval_controlled.py \
    --features results/bitstream_features_controlled.csv --model xgb

# 4. Wang & Farid (2006) 像素基线在受控数据上的重跑 —— 替换旧稿 0.503
python scripts/evaluate_su2018_blind.py \
    --index data_controlled/sample_index.csv \
    --output results/su2018_blind_controlled.csv
    # 运行结束会打印 "AUC: x.xxxx" —— 这就是受控值, 用来替换论文里的 0.503

# 5. (可选) 近参数子集研究 (只留 |ΔQP| ≤ 3)
python scripts/eval_controlled.py \
    --features results/bitstream_features_controlled.csv --max-delta 3 --model xgb
```

## 跑完之后，论文要更新的两处数字

| 位置 | 旧值 (非受控/随机 CV) | 新值 (来自上面) |
|---|---|---|
| 摘要 / 主表 / Baseline 表 的 Wang & Farid 基线 | `0.503` | 步骤 4 打印的受控 AUC |
| 摘要 / 贡献句 / 主表的码流域头条 | `0.991`(随机 CV) / `0.9997` | 步骤 3 的 **source-aware** 受控 AUC |

> ⚠️ 关键：步骤 3 的 **source-aware** 数字才是诚实头条，取代原来虚高的随机
> 5-fold CV (0.99)。两个方法都基于**同一套受控双臂协议 / 同一 20 个源**，但像素基线
> 只跑了 120-clip 平衡子集（每源×码器 1 个 double 变体，因像素解码昂贵），
> 而码流域分类器用满 300-clip；故两者是**同协议、不同子集**的定性比较，论文已如实说明。

## 复用提示
- `evaluate_su2018_blind.py` 已支持 `--index/--output/--max/--dry-run`，
  且受控 index 多出的 `q1/q2/delta_qp` 列会自动透传到输出 CSV。
- 旧基线结果仍在 `results/su2018_blind_evaluation.csv`（非受控数据，保留作对照）。
- 若本机无 `PyAV`，步骤 2/4 会报 import 错；装法：`pip install av` (需系统 ffmpeg)。
- 步骤 4 还需要 `cv2` (opencv-python)：`pip install opencv-python`。
- 受控语料规模不一致提示：`sample_index.csv` 是 300 clip (每源×码器 1 single + 4 double)，
  但 `su2018_blind_controlled.csv` 只有 120 行 (每源×码器 1 single + 1 double)。
  论文里的 Wang & Farid 基线 0.566 是在这 120-clip 平衡子集上算出的；若想复现该数，
  步骤 4 加 `--max 120` (或只对子集跑)。
- `prepare_samples.py` 的路径已**参数化**（默认仍用本机写死值，行为不变）：
  换机器/换数据根目录时只传参即可，无需改代码：
  ```bash
  python scripts/prepare_samples.py \
      --video-dir  /path/to/source_videos \
      --data-dir   /path/to/blind_detection/data_controlled \
      --ffmpeg     /usr/bin/ffmpeg \
      --duration   30 \
      --seed       42
  ```
  - `--ffmpeg` 未给且默认镜像路径不存在时，会**自动回退到 PATH 上的 `ffmpeg`** 并打印提示；两者都没有才告警。
  - `--video-dir` / `--data-dir` / `--duration` / `--seed` 同理，均带默认值，省略即沿用原行为。

## 可复现性（任意机器一键复现）
- 三步脚本的默认路径现已**全部相对脚本自身位置解析**（scripts/ → 上溯到 blind_detection/），
  不再含本机绝对路径。换机只需：
  1) 进入项目根目录 `blind_detection/`；
  2) step 1 用 `--video-dir` 指定你机器上的源视频目录
     （其余 `--data-dir` / `--ffmpeg` / `--duration` / `--seed` 均可省略，沿用默认）；
  3) step 2 / step 4 即便不传 `--index` / `--output`，
     也会自动落到 `data_controlled/sample_index.csv` 与 `results/*.csv`。
- `evaluate_su2018_blind.py` 里的 `su2018_baseline` import 也靠相对上溯到
  blind_detection 的父目录（即 `su2018_baseline.py` 所在处）注入 `sys.path`，换机同样无需改。
- `eval_controlled.py` 本身无硬编码路径，只靠 `--features` 入参。
- 最简调用（从 blind_detection/ 根目录）：
  ```bash
  python scripts/prepare_samples.py --video-dir /your/source_videos
  python scripts/extract_bitstream_expanded.py
  python scripts/eval_controlled.py --features results/bitstream_features_controlled.csv --model xgb
  python scripts/evaluate_su2018_blind.py --dry-run   # 仅统计, 不需 av/cv2
  ```
