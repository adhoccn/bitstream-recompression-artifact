# Fig. 2 — 特征重要性（fig_importance）

稿件位置：`paperB_body.tex` 的 `\label{fig:importance}`，正文按 `Fig. 2` 引用。
Caption：*Feature importance. Left: XGBoost gain (top-10). Right: standardized
logistic-regression weights (top-10); both rank packet-size statistics first, qualitatively
supporting the mechanism described above.*

两个模型都把**包大小统计**排在最前，用来支持正文 §Analysis 讲的机制
（再压缩留下的痕迹体现在码率分配/包大小上，而不是帧结构）。

---

## 文件

| 文件 | 说明 |
|---|---|
| `fig_importance.png` | **稿件里的正式图**（148,164 B，2639×1304）—— **即本脚本的重跑产物**，2026-10-11 已替换 |
| `gen_fig_importance.py` | 生成代码（原 `redraw_figs.py`，一脚本出两图） |
| （换图前的旧图） | 已存档：`en_condensed/_archive_20261011/figs_before_rerun_20261011/fig_importance__figs.png`（153,721 B） |

## 重跑

```bash
cd figures/fig_importance
python gen_fig_importance.py      # 主产物 fig_importance.png
                                  # 副产物 _also_fig_scores.png（同一脚本顺带画的 Fig.3）
```

依赖：`numpy`、`pandas`、`seaborn`、`matplotlib`、`scikit-learn`、`xgboost`。

### 数据来源（按此顺序查找）

1. `../../experiment/bitstream_features_controlled.csv`（**交付包内**，191 KB，随包提供）
2. 找不到则回退原始绝对路径
   `C:/baidunetdiskdownload/lightweight/blind_detection/results/bitstream_features_controlled.csv`

### 随机性

脚本里的模型**固定了种子**：CV 用 `seed in range(N_SEEDS)`，最终模型
`XGBClassifier(random_state=42)`、`LogisticRegression(random_state=42)`。
所以同一个环境下重跑是确定的；跨环境仍可能有差异（见下）。

## ✅ 重跑验证（2026-10-11 实测，稿件已换成重跑产物）

**逐位可复现**：同一环境连跑两次，md5 完全一致 ⇒ 仓库里这张图就是脚本的输出。

| | 字节数 | md5 |
|---|---|---|
| 本脚本重跑（= 稿件现用图） | 148,164 | `4df7a773243aea0dbdcfea258d1ce4c0` |
| 换图前的旧图（已存档） | 153,721 | `e494115c28b71eb35b29e34f0fcda7c1` |

旧图与新图**字节不同、视觉等价**：平均像素差 6.58 / 255，差异 > 30 的像素占 4.07%，
尺寸 2653×1303 → 2639×1304。差异来源为渲染环境，非数据或逻辑
（判据见 `../fig_crosscodec_heatmap/README.md`：那张图数据硬编码，照样字节不同）。

脚本运行时的标准输出会打出 top 特征，可用于**数值级**比对（比像素比对更可靠）——
新旧两次运行**完全一致**：

```
XGB top = pts_delta_std (0.071), LR top = pkt_size_p25 (0.521)
```
