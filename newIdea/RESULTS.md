# IP-IQA + 多尺度双注意力局部质量分支 —— 正式实验结果

> **方法**：IP-IQA（跨模态主干）+ Local Quality Branch：
> ① Multi-Scale Local Representation（fine 16×16 + coarse 8×8）
> ② Spatial-Channel Dual Attention（ChannelBlock + SpatialBlock，两尺度各一套）
> ③ Importance-Aware Patch Weighted Aggregation（patch score + weight → q_local）
> 与 q_base 门控融合。已移除 Cross-Branch Attention（见 `方案调整/完整消融方案.md`）。

> **正式协议**（所有消融/正式结果统一）：
> AGIQA-3K（`splits/seed42.json`，2383/599）或 AIGCIQA2023（`splits/aigciqa2023_seed42.json`，按 prompt 80/20，1920/480）
> 单阶段训练（从 CLIP 初始化，**无 warm-start staging**）
> LR：`linear_warmup_cosine_lr`（约 1 epoch warmup → cosine 到 1e-6）+ 分组 LR（backbone 1× / local 3× / 新模块 10× / gates 10×）
> 100 epoch、batch 32、512×512、AMP
> **checkpoint 选择：best-quality = argmax(SRCC_qual + PLCC_qual)**（方案第 37 节）
> 日期：2026-08-19/20，RTX 3090 24GB，torch 2.3.1+cu121

---

## 一、AGIQA-3K 主消融表（best-quality，100 epoch）

| 行 | 模型 | Multi-Scale | Dual Attention | Patch Weight | best-ep | SRCC ↑ | PLCC ↑ | KROCC ↑ | align SRCC |
|---|---|---|---|---|---|---|---|---|---|
| A0 | IP-IQA baseline | × | × | × | 64 | 0.8261 | 0.8835 | 0.6419 | 0.6318 |
| A1 | + Multi-Scale | ✓ | × | × | 52 | 0.8269 | 0.8858 | 0.6428 | 0.6269 |
| A2 | + Dual Attention | ✓ | ✓ | × | 62 | **0.8360** | **0.8900** | **0.6499** | **0.6454** |
| A3 | **Full Ours** | ✓ | ✓ | ✓ | 59 | **0.8372** | 0.8886 | **0.6529** | 0.6326 |

### 差值（同一协议）

| 对比 | SRCC Δ | PLCC Δ | KROCC Δ |
|---|---|---|---|
| A1 − A0（Multi-Scale 贡献） | +0.0008 | +0.0023 | +0.0009 |
| A2 − A1（Dual Attention 贡献） | **+0.0091** | **+0.0042** | **+0.0071** |
| A3 − A2（Patch Weight 贡献） | +0.0012 | −0.0014 | +0.0030 |
| **A3 − A0（总体）** | **+0.0111** | **+0.0051** | **+0.0110** |

### 诚实归因（对应方案第 43 节预警的情况）

- **总体有效**：Full Ours 相对 IP-IQA baseline 提升 **+0.0111 SRCC / +0.0051 PLCC / +0.0110 KROCC**，且三指标同向。
- **主要贡献来自创新点 2（双注意力）**：A2−A1 = +0.0091 SRCC，是增益的绝对主力。
- **创新点 1（多尺度）与创新点 3（patch 加权）单独几乎持平**：A1−A0 ≈ +0.001、A3−A2 ≈ +0.001（PLCC 甚至略降）。
  因此不应写成"三步各自单调上涨"，而应表述为：
  > 多尺度双注意力局部增强（MS+DA）是性能提升的核心来源；patch 加权聚合在双注意力之上保持稳定、贡献有限。

## 二、AIGCIQA2023（第二数据集，Full Ours）

| 模型 | best-ep | SRCC ↑ | PLCC ↑ | KROCC ↑ | align SRCC |
|---|---|---|---|---|---|
| A3 Full Ours | 79 | 0.8250 | 0.8470 | 0.6132 | 0.6339 |

- 说明：AIGCIQA2023 按 prompt 分组 80/20（1920/480），未在其上跑消融（消融只在 AGIQA-3K）。
- 数值处于文献竞争区间：DCMPLN Table 1 中 RichIQA 0.8245 / CLIP-AGIQA 0.8324（注意协议不同：224 输入、batch64、50ep、10 runs，见 `方案调整/总实验方案.md`）。

## 三、文献对比参考（literature-reported，非我们复现）

> 数值来自 DCMPLN 2026 Table 1（Displays, 91, 103208），协议与本项目不同，仅作参考定位。

| Method | AGIQA-3K SRCC | AGIQA-3K PLCC | AIGCIQA2023 SRCC | AIGCIQA2023 PLCC |
|---|---:|---:|---:|---:|
| HyperIQA | 0.8355 | 0.8903 | 0.8174 | 0.8459 |
| RichIQA | 0.8592 | 0.8976 | 0.8245 | 0.8564 |
| IP-IQA (published, 含 Image2Prompt 预训练) | 0.8634 | 0.9116 | — | — |
| CLIP-AGIQA | 0.8747 | 0.9190 | 0.8324 | 0.8604 |
| **Ours (A3, 100ep, 无 Image2Prompt)** | **0.8372** | **0.8886** | **0.8250** | **0.8470** |

> 注意：IP-IQA published 含 Image2Prompt 额外预训练，我们没有做；与文献横比应标注协议差异，真正公平的对比是上表一（同协议内部消融）。

## 四、运行记录（run 目录 ↔ 结果）

| 任务 | run 目录 | tag |
|---|---|---|
| AGIQA-3K A0 | `run/20260820002_A0_ipqa` | A0_ipqa |
| AGIQA-3K A1 | `run/20260820011_A1_ms` | A1_ms |
| AGIQA-3K A2 | `run/20260820015_A2_ms_da` | A2_ms_da |
| AGIQA-3K A3 | `run/20260820024_A3_full` | A3_full |
| AIGCIQA2023 A3 | `run/20260820033_E3_aigciqa2023` | E3_aigciqa2023 |

每个 run 目录含 `log.txt`（每 epoch 全量指标 + outer/refine gate）、`checkpoint_best.pth`（best-quality）、`checkpoint_latest.pth`。
串行训练日志：`run/serial_train.log`。

## 五、复现命令

```bash
cd /root/autodl-tmp/newIdea_extracted/newIdea/ipqa_local
# 一键串行（AGIQA-3K A0-A3 + AIGCIQA2023 A3）
bash serial_train.sh
# 或单独跑某一项
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_ablation_A0.yaml --seed 42 --num_cv 1
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_ablation_A1_ms.yaml --seed 42 --num_cv 1
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_ablation_A2_ms_da.yaml --seed 42 --num_cv 1
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_ablation_A3_full.yaml --seed 42 --num_cv 1
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_aigciqa2023_msda.yaml --seed 42 --num_cv 1
```

## 六、Development Record（旧协议 quick 结果，仅作参考，不进正式表）

早期协议：constant LR、部分 warm-start 权重、20/30 epoch、best 按 joint agg 选。

| 实验 | 协议 | SRCC | PLCC |
|---|---|---|---|
| E0 IP-IQA | 20ep bs8 | 0.8153 | 0.8744 |
| E1 +Weighted Local | 20ep bs8 | 0.8262 | 0.8816 |
| E3 MSDA 冷启动 | 20ep bs32 | 0.8172 | 0.8759 |
| E3-v2 MSDA warm-start | 30ep bs32 | 0.8308 | 0.8839 |
| E1-continue 对照 | 30ep bs32 无 MSDA | 0.8188 | 0.8816 |
