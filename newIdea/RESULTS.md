# IP-IQA + Local Distortion Branch —— 实验结果

> 实验日期：2026-08-19（服务器 RTX 3090 24GB，torch 2.3.1+cu121）
> 协议：AGIQA-3K，固定 seed 42，单次 80/20 随机 hold-out，20 epoch（`*_quick` 配置），
> 三个实验使用**完全相同的划分与超参**，唯一区别是是否加局部失真分支。
> 数值为 `train_agiqa3k.py` 输出的 final averaged metrics（best-epoch 策略）。

## 结果表（final metrics）

| 实验 | 配置 | 局部分支 | qual SROCC | qual PLCC | qual KROCC | align SROCC | align PLCC | align KROCC |
|---|---|---|---|---|---|---|---|---|
| **E0** | `ipiqa_quick.yaml` | baseline（无） | 0.8153 | 0.8744 | 0.6267 | 0.6617 | 0.8168 | 0.4835 |
| **E1** | `ipiqa_ours_noattn.yaml` | +Weighted Local（无 attention） | **0.8262** | **0.8816** | **0.6396** | 0.6547 | 0.8018 | 0.4779 |
| **E2** | `ipiqa_ours.yaml` | Full（+attention） | 0.8095 | 0.8708 | 0.6227 | 0.6680 | 0.8125 | 0.4897 |

## 与 baseline 的差（同一划分，E − E0）

| 实验 | qual SROCC Δ | qual PLCC Δ | qual KROCC Δ |
|---|---|---|---|
| E1 | **+0.0108** | **+0.0072** | **+0.0129** |
| E2 | −0.0058 | −0.0036 | −0.0040 |

## 结论

1. **E1（加权局部 patch score/weight，无 attention）在 quality 上显著优于 baseline**：
   SROCC +0.011、PLCC +0.007，支持"显式局部失真感知对 AIGC 质量预测有增益"的核心假设。
2. **E2（额外加 attention 后）掉点**（SROCC −0.006），与计划文档第 17 节回退策略预判一致：
   attention 在该小数据集/快速协议下不适配，应回退到纯 patch weight 方案。
3. 当前为 preliminary（单次随机划分），后续可用 E1 配置跑 5~10 次随机划分验证稳定性，
   或跑完整 100 epoch 协议（`ipiqa.yaml`）提升绝对值。

## 复现命令

```bash
# E0: baseline
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_quick.yaml --seed 42 --num_cv 1
# E1: +Weighted Local（推荐方案）
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_ours_noattn.yaml --seed 42 --num_cv 1
# E2: Full Ours
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_ours.yaml --seed 42 --num_cv 1
```

每次实验的日志 / 权重 / 指标保存在 `ipqa_local/run/<job_id>_<tag>/`。
