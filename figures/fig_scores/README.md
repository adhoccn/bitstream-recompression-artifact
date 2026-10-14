# Fig. 3 — 分数分布（fig_scores）

稿件位置：`paperB_body.tex` 的 `\label{fig:scores}`，正文按 `Fig. 3` 引用。
Caption：*Score distributions of the double (positive) and single (negative) classes under
source-aware cross-validation (out-of-fold XGBoost scores). The double class is right-skewed,
concentrating recall at low false-positive rates; the absolute score scale varies across
sources, so thresholds do not transfer.*

这张图是**"阈值不能跨源迁移"**这条阴性发现的视觉证据：分布看着分得开，但绝对分数尺度
随源而变 ⇒ 在训练源上标定的阈值迁到未见源会崩（实测目标 FPR 0.05、实际 0.75）。

---

## 文件

| 文件 | 说明 |
|---|---|
| `fig_scores.png` | **稿件里的正式图**（76,497 B，1743×1083）—— **即本脚本的重跑产物**，2026-10-11 已替换 |
| `gen_fig_scores.py` | 生成代码（原 `redraw_figs.py`，一脚本出两图） |
| （换图前的旧图） | 已存档：`en_condensed/_archive_20261011/figs_before_rerun_20261011/fig_scores__figs.png`（71,997 B） |

## 重跑

```bash
cd figures/fig_scores
python gen_fig_scores.py          # 主产物 fig_scores.png
                                  # 副产物 _also_fig_importance.png（同一脚本顺带画的 Fig.2）
```

依赖：`numpy`、`pandas`、`seaborn`、`matplotlib`、`scikit-learn`、`xgboost`。

### 数据来源（按此顺序查找）

1. `../../experiment/bitstream_features_controlled.csv`（**交付包内**，191 KB）
2. 找不到则回退原始绝对路径
   `C:/baidunetdiskdownload/lightweight/blind_detection/results/bitstream_features_controlled.csv`

### 随机性

种子固定（`range(N_SEEDS)` 做 CV，最终模型 `random_state=42`），同环境重跑确定。

## ✅ 重跑验证（2026-10-11 实测，稿件已换成重跑产物）

**逐位可复现**：同一环境连跑两次，md5 完全一致 ⇒ 仓库里这张图就是脚本的输出。

| | 字节数 | md5 |
|---|---|---|
| 本脚本重跑（= 稿件现用图） | 76,497 | `b214058e48fe8f6e99ae8666513ff175` |
| 换图前的旧图（已存档） | 71,997 | `497e4586f2ccf0f6dee58d026b941965` |

旧图与新图**尺寸完全一致**（1743×1083），平均像素差 **1.45 / 255**，
差异 > 30 的像素仅 **1.19%** ⇒ 差异基本是字体抗锯齿级别。

脚本运行时的标准输出可用于数值级比对——新旧两次运行**完全一致**：

```
fig_scores.png: 0.748 (single mean) vs 0.828 (double mean)
```
