# 复现说明（Paper B）

论文结论是**阴性**结论，"能复现"因此是这篇稿的立身之本。本文给出从零跑通的最短路径。

---

## 1. 环境

| 项 | 要求 |
|---|---|
| Python | 3.13（Windows 实测 3.13.12；其余版本应可行，未测） |
| ffmpeg | **必须**。缺 ffmpeg 时特征提取静默失败，症状看着像"素材坏了" |
| Python 包 | `av`(PyAV)、`numpy`、`pandas`、`scipy`、`scikit-learn`、`xgboost`、`matplotlib`、`seaborn` |

```bash
pip install -r requirements.txt
```

> `requirements.txt` 里的版本是 2026-10-12 实测跑通的那一套。
> **matplotlib 换版本会改变图的字节**（字体度量不同），要逐位复现 `figures/` 的图
> 须用 `matplotlib==3.10.9`；换版本不影响任何数字。

作图另需中文字体无关，但 **Times New Roman** 缺失会让图的度量变化（见第 4 节）。

---

## 2. 三步流水线

以下命令在 **`paperB/` 根目录**执行（源码在 `experiment/`）：

```bash
# (1) 构造受控双臂样本：single 与 double 同编码器、同素材（需原始视频 + ffmpeg）
py experiment/prepare_samples.py --max 20

# (2) 抽 39 维比特流特征（不解码到像素、不用 GPU）
py experiment/extract_bitstream_expanded.py \
      --index  data_controlled/sample_index.csv \
      --output results/bitstream_features_controlled.csv

# (3) 受控协议评估：source-aware 分组 CV + 跨编码器矩阵 + 码率混淆消融
py experiment/eval_controlled.py --features results/bitstream_features_controlled.csv
```

### 只有特征表也能复现（2026-10-12 实测）

前两步需要原始视频；仓库自带特征表，所以离线也能验证论文数字：

```bash
py experiment/eval_controlled.py \
      --features experiment/bitstream_features_controlled.csv
py experiment/eval_controlled.py --self-test    # 合成数据检验协议，零素材依赖
```

`eval_controlled.py` 常用开关：

| 开关 | 作用 |
|---|---|
| `--model xgb` / `--model lr` | 换分类器 |
| `--max-delta 3` | 只看近参数（double 的 ΔCRF ≤ 3）子集 |
| `--self-test` | 用合成数据自测协议本身（不依赖任何素材） |

默认路径相对脚本位置解析（`../data_controlled`、`../results`），
所以从项目根目录直接跑即可，无需改路径。

---

## 3. 会拿到什么（论文里的数字）

| 项 | 数字 | 出处 |
|---|---|---|
| source-aware CV 的 AUC | $0.665 \pm 0.012$（300 clip，H.264/H.265/VP9） | 摘要 |
| 与像素基线在匹配 120 clip 子集上 | $0.536 \pm 0.018$ vs $0.566$，配对 bootstrap $p\approx0.8$ | 摘要 |
| 训练源上标定的阈值迁到未见源 | 目标 FPR 0.05，实测 **0.75** | 摘要 |
| 两遍压缩趋于一致时 | AUC 降到 $0.574$（$\Delta\mathrm{CRF}\le3$） | 摘要 |
| 跨编码器迁移 | $0.46$–$0.59$ | 摘要 |

**这些数字的作用是"未检测到差异"，不是"检测到差异"。** 复现时若得到更强的数字，
先怀疑协议被污染（最常见的是源泄漏：同一原始视频的不同片段被分到训练与测试集）。

---

## 4. 图：逐位可复现（稿件里的图就是脚本输出）

`figures/` 下每张图都有 `gen_*.py`，在本目录里直接跑即可产出同名 PNG，
**与稿件所用图逐字节相同**：

```bash
cd figures/fig_scores && python gen_fig_scores.py     # -> fig_scores.png (76,497 B)
```

验证（2026-10-11 实测，`src/check/verify_determinism.py`）：每个脚本在临时目录连跑两次，
两份输出 md5 一致，三张图全部通过。

| 图 | 稿件现用图（= 重跑产物） | 换图前的旧图（已存档） | 旧↔新差异 |
|---|---|---|---|
| fig_scores | 76,497 B `b214058e…` | 71,997 B | 1.45 / 255（1.19% 像素 > 30） |
| fig_importance | 148,164 B `4df7a773…` | 153,721 B | 6.58 / 255（4.07%） |
| fig_crosscodec_heatmap | 76,417 B `565d5d39…` | 72,624 B | 10.72 / 255（7.46%） |

**旧图与新图为何字节不同**：`fig_crosscodec_heatmap` 的数值是脚本里**硬编码**的矩阵、
零随机性，它照样字节不同 ⇒ 差异只来自渲染环境（字体度量、matplotlib 版本），
与数据、模型、结论无关。换图只影响排版像素，**论文里任何数字都没变**
（脚本 stdout 的 `0.748 vs 0.828`、`XGB top = pts_delta_std (0.071)` 新旧完全一致）。

---

## 5. 归档取 DOI（建议做）

IEEE Access 鼓励可复现研究。建议把本包（代码 + `bitstream_features_controlled.csv`
+ 三张图）整体归档到 **IEEE DataPort** 或 **Zenodo** 取 DOI，然后在
`CITATION.cff` 里回填 `identifiers`，并在论文的 artifact 引用处写真实 DOI。

⚠ DataPort 上传时**务必选 Open Access**：否则非会员的审稿人下载不到，
归档等于没做。
