# Fig. 1 — 跨编码器迁移热力图（fig_crosscodec_heatmap）

稿件位置：`paperB_body.tex` 的 `\label{fig:crosscodec}`，正文按 `Fig. 1` 引用。
Caption：*Cross-codec transfer (XGBoost, controlled corpus). Each cell reports AUC ± 95% CI.
Diagonal: leave-source-out CV within each codec (10 seeds × 5 folds). Off-diagonal: train on
one codec, test on another; mean over 10 seeds with subsampled XGBoost.*

这张图是论文**阴性结论的主要证据之一**：对角线（同编码器）尚可，非对角线（跨编码器）
掉到 0.46–0.59 ⇒ 双量化指纹不是编码器无关的。

---

## 文件

| 文件 | 说明 |
|---|---|
| `fig_crosscodec_heatmap.png` | **稿件里的正式图**（76,417 B，1022×845）—— **即本脚本的重跑产物**，2026-10-11 已替换 |
| `gen_fig_crosscodec_heatmap.py` | 生成代码（57 行，**数据硬编码在脚本里**） |
| （换图前的旧图） | 已存档：`en_condensed/_archive_20261011/figs_before_rerun_20261011/fig_crosscodec_heatmap__figs.png`（72,624 B） |

## 重跑

```bash
cd figures/fig_crosscodec_heatmap
python gen_fig_crosscodec_heatmap.py
```

依赖：`numpy`、`matplotlib`。**不需要任何数据文件**——3×3 的 AUC 矩阵与 caption 文本
都写在脚本里，来源 `rebuttal/make_crosscodec_fig.py`。

输出路径已改为脚本所在目录（原脚本会同时写到 4 个位置，含
`C:/baidunetdiskdownload/...`，交付版只写本目录）。

## ✅ 重跑验证（2026-10-11 实测，稿件已换成重跑产物）

**逐位可复现**：同一环境连跑两次，md5 完全一致 ⇒ 仓库里这张图就是脚本的输出。

| | 字节数 | md5 |
|---|---|---|
| 本脚本重跑（= 稿件现用图） | 76,417 | `565d5d3958acf3e7b1f075cff104d093` |
| 换图前的旧图（已存档） | 72,624 | `e42541549ee08534471d0faa737fe4f9` |

旧图与新图**字节不同、视觉等价**：平均像素差 10.72 / 255，差异 > 30 的像素占 7.46%，
尺寸 1005×845 → 1022×845（宽度差 17 px）。

差异来源判据（这条最有力）：**本图数据完全硬编码、脚本无任何随机性**，新旧照样字节不同
⇒ 差异只能来自**渲染环境**（字体度量 / matplotlib 版本），与数据、模型、结论无关。
换图只影响排版像素，**论文里的任何数字都没有变化**。
