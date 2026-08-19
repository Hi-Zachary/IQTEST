# IP-IQA + Local Distortion Branch：详细实施与实验方案

> 目标：在一张 RTX 3090 24GB 上，用 AGIQA-3K 做一个 **今晚可以先跑出 preliminary result** 的版本。
>
> 当前优先级：**先保证能跑、能和 reproduced baseline 公平比较，再追求完整论文协议。**

---

# 1. 方法总览

## 1.1 输入与输出

输入：

```text
AI-generated image I
+
generation prompt T
```

输出继续沿用 IP-IQA 的 AGIQA-3K 双输出：

```text
quality score    -> perceptual quality MOS
alignment score  -> image-text alignment MOS
```

我们只修改 `quality score`；`alignment score` 尽量保持 IP-IQA 原结构。

---

## 1.2 原 IP-IQA

原 IP-IQA 的主干可以简化为：

```text
                         Image
                           ↓
                      CLIP RN50
                           ↓
                  spatial feature F
                    /             \
                   /               \
                  ↓                 ↓
        CLIP visual pooling   TextAttentionPool2d
                  ↓                 ↑
          global_visual        Prompt feature
                   \               /
                    \             /
                     concatenate
                         ↓
                       MLP
                         ↓
              [quality, alignment]
```

IP-IQA 源码接口：

```python
# ipiqa/models/ipiqa.py

txt_feat = self.encode_text(text)
feat = self.resnet50(x)

global_visual = self.attnpool(feat)
global_txt = self.txt_attnpool(feat, txt_feat)

global_feat = torch.cat([global_visual, global_txt], dim=-1)
output = self.head(global_feat)
```

这里的 `feat` 是我们最重要的缝合接口。

对于默认 `input_resolution=512`，CLIP RN50 的最终 spatial feature 可以按约：

```text
F: [B, 2048, 16, 16]
```

来设计局部分支。

---

# 2. 我们增加的 Local Distortion Branch

## 2.1 结构

建议第一版尽量克制：

```text
F: [B, 2048, 16, 16]
        ↓
1×1 Conv projection
2048 -> 256
        ↓
[B, 256, 16, 16]
        ↓
flatten spatial dimension
[B, 256, 256]
        ↓
1 × lightweight transpose-attention block
        ↓
[B, 256, 256]
        ↓
transpose as 256 patches
[B, 256, 256]
        ↓
   ┌─────────────┬─────────────┐
   ↓             ↓
patch score   patch weight
   ↓             ↓ sigmoid
   └────── × ────┘
          ↓
weighted average
          ↓
       q_local
```

为什么先 `2048 -> 256`：

1. 降低 attention 和 MLP 计算量；
2. 避免把 AM-BQA 的完整大模块硬搬过来；
3. 3090 更容易控制在合理显存；
4. 保留清楚的网络模块感。

---

## 2.2 Attention 模块参考

参考 AM-BQA：

```text
models/ambqa.py
models/ambqa_withart.py
```

第一晚版本建议只保留一个简单的 `TABlock` 风格模块，不要搬 ART/Swin/完整多阶段网络。

可以新增：

```text
ipiqa/models/local_distortion.py
```

示例：

```python
import torch
import torch.nn as nn


class TransposeAttentionBlock(nn.Module):
    """Lightweight adaptation inspired by AM-BQA TABlock."""
    def __init__(self, spatial_dim=256, drop=0.1):
        super().__init__()
        self.q = nn.Linear(spatial_dim, spatial_dim)
        self.k = nn.Linear(spatial_dim, spatial_dim)
        self.v = nn.Linear(spatial_dim, spatial_dim)
        self.scale = spatial_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, C, N]
        residual = x
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)
        out = self.drop(out)
        return out + residual
```

注意：这是 **根据 AM-BQA TABlock 思想适配到 IP-IQA feature shape 的轻量重写**，不是声称逐行复现 AM-BQA 完整网络。

---

## 2.3 Patch score / weight head

继续在 `local_distortion.py`：

```python
class LocalDistortionBranch(nn.Module):
    def __init__(self, in_channels=2048, hidden_dim=256,
                 spatial_dim=256, drop=0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        self.attn = TransposeAttentionBlock(
            spatial_dim=spatial_dim,
            drop=drop,
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1),
        )

        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        # feat: [B, 2048, H, W]
        x = self.proj(feat)               # [B, 256, H, W]
        B, C, H, W = x.shape

        x = x.flatten(2)                   # [B, C, N]
        x = self.attn(x)                   # [B, C, N]
        x = x.transpose(1, 2).contiguous() # [B, N, C]

        patch_score = self.score_head(x)   # [B, N, 1]
        patch_weight = self.weight_head(x) # [B, N, 1]

        q_local = (patch_score * patch_weight).sum(dim=1) / \
                  (patch_weight.sum(dim=1) + 1e-6)

        return q_local
```

这里没有在 `score_head` 最后一层加 ReLU，是我们针对 AGIQA 回归做的工程调整，避免无必要地限制输出范围。

---

# 3. 和 IP-IQA 具体怎么融合

修改：

```text
ipiqa/models/ipiqa.py
```

## 3.1 `__init__` 中加入

```python
from ipiqa.models.local_distortion import LocalDistortionBranch
```

然后：

```python
self.use_local_branch = True

spatial_side = input_resolution // 32
self.local_branch = LocalDistortionBranch(
    in_channels=2048,
    hidden_dim=256,
    spatial_dim=spatial_side * spatial_side,
    drop=0.1,
)

# 初始更相信原 IP-IQA，降低一上来掉点风险
self.local_gate_logit = nn.Parameter(torch.tensor(-2.0))
```

`sigmoid(-2) ≈ 0.119`，也就是初始大约 88% 原 IP-IQA + 12% local branch。

---

## 3.2 修改 forward

原来：

```python
return self.head(global_feat)
```

改成：

```python
base_output = self.head(global_feat)  # [B, 2]

q_base = base_output[:, 0:1]
align = base_output[:, 1:2]

q_local = self.local_branch(feat)

g = torch.sigmoid(self.local_gate_logit)
q_final = (1.0 - g) * q_base + g * q_local

output = torch.cat([q_final, align], dim=1)
return output
```

这版的优点：

- 原 alignment 完全不动；
- 原 quality path 保留；
- local branch 是补充分支，不是替换；
- gate 可以自己学习最终应该信 local 多少。

---

# 4. 为什么不直接把两个分数相加

不建议第一版：

```python
q_final = q_base + q_local
```

因为两个分支初期尺度可能不同。

也不建议直接删掉原 IP-IQA quality head。

最稳妥的是：

```text
Original quality prediction
            +
Local quality prediction
            ↓
learnable gate
```

这样即使 local branch 暂时没有学好，模型仍然可以主要依赖原路径。

注意：这只能 **降低掉点风险**，不能保证一定提升。

---

# 5. 数据集：AGIQA-3K

## 5.1 下载源码/标注

```bash
git clone https://github.com/lcysyzxdxc/AGIQA-3k-Database.git
```

仓库：

https://github.com/lcysyzxdxc/AGIQA-3k-Database

`data.csv` 已经在 repo 里。

---

## 5.2 下载图片

Google Drive：

https://drive.google.com/file/d/1ObuOZ6YZqZuxe4oRlaf3kdOBlTRg2GE4/view?usp=sharing

夸克：

https://pan.quark.cn/s/10187e65d5c1

下载后解压。

建议最终目录：

```text
IP-IQA/
├── cache/
│   ├── ckpt/
│   │   └── clip/openai/resnet/RN50.pt
│   │
│   └── data/
│       └── aigc_qa_3k/
│           ├── AGIQA-3K/
│           │   ├── AttnGAN_normal_000.jpg
│           │   ├── ...
│           │   └── xxx.jpg
│           ├── data.csv
│           └── mos_joint.xlsx
```

---

# 6. 生成 IP-IQA 需要的 `mos_joint.xlsx`

AGIQA-3K 官方 `data.csv` 列是：

```text
name,prompt,adj1,adj2,style,mos_quality,std_quality,mos_align,std_align
```

而 IP-IQA 的 `AGIQA3k` Dataset 实际只要求前四列按下面顺序：

```text
image_name
prompt
quality_mos
alignment_mos
```

可以自己生成：

```python
import pandas as pd

src = "AGIQA-3k-Database/data.csv"
out = "IP-IQA/cache/data/aigc_qa_3k/mos_joint.xlsx"

df = pd.read_csv(src)

df2 = df[["name", "prompt", "mos_quality", "mos_align"]].copy()
df2.columns = ["name", "prompt", "mos_quality", "mos_align"]

df2.to_excel(out, index=False)
print("saved:", out, len(df2))
```

把图片复制/解压到：

```text
IP-IQA/cache/data/aigc_qa_3k/AGIQA-3K/
```

---

# 7. 下载 CLIP RN50

直链：

https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt

Linux 可直接：

```bash
mkdir -p cache/ckpt/clip/openai/resnet
wget -O cache/ckpt/clip/openai/resnet/RN50.pt \
  https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
```

---

# 8. 环境安装

IP-IQA 官方推荐：

```bash
conda create -n ipiqa python=3.9
conda activate ipiqa
pip install -e .
```

如果 `openpyxl` 缺失（生成 xlsx 时）：

```bash
pip install openpyxl
```

---

# 9. 今晚的单次随机划分方案

## 9.1 不需要自己重新写 split

IP-IQA 官方 `train_agiqa3k.py` 已经包含：

```python
count = int(0.8 * 300)
indices = np.random.permutation(300)
```

然后根据文件名最后的 object/content ID，把 300 个内容 ID 中：

```text
240 -> train
60  -> hold-out eval
```

同一个 ID 对应的图片会整体进入同一侧。

因此今晚直接固定：

```text
seed = 42
num_cv = 1
```

即可得到一个可重复的单次 80/20 随机 hold-out。

运行：

```bash
python train_agiqa3k.py \
  --cfg-path ipiqa/projects/agiqa3k/ipiqa_quick.yaml \
  --seed 42 \
  --num_cv 1
```

### 对结果的正确表述

明天面试说：

> “目前采用固定 seed 的单次 80/20 hold-out 做 preliminary experiment，baseline 和改进模型完全使用同一划分；后续再补多次随机划分验证稳定性。”

不要说：

> “这是 AGIQA-3K 官方固定测试集。”

因为数据集没有唯一固定 test set。

---

# 10. Quick Config：今晚版超参数

建议复制：

```text
ipiqa/projects/agiqa3k/ipiqa.yaml
```

为：

```text
ipiqa/projects/agiqa3k/ipiqa_quick.yaml
```

推荐第一晚：

```yaml
model:
  arch: "ipiqa"
  base_ckpt: cache/ckpt/clip/openai/resnet/RN50.pt
  input_resolution: 512
  output_dim: 2
  load_finetuned: False
  head_scale: 10.
  use_mlp_head: True
  dropout_rate: 0.5
  freeze_text: True
  qa_token: True

dataset:
  data_path: "cache/data/aigc_qa_3k/mos_joint.xlsx"
  vis_root: "cache/data/aigc_qa_3k/AGIQA-3K/"
  transform_train:
    name: "image_train_processor"
    cfg:
      image_size: 512
  transform_val:
    name: "image_eval_processor"
    cfg:
      image_size: 512

run:
  task: "agiqa_doublescore"
  resume_ckpt_path: null

  lr_sched: "constant_lr"
  lr_decay_rate: null
  warmup_lr: -1
  warmup_steps: 0
  min_lr: 1e-5

  init_lr: 1e-5
  lr_layer_decay: 1
  weight_decay: 0
  beta2: 0.999

  batch_size: 8
  batch_size_val: 16
  num_worker: 8

  # 今晚先 20；如果明显很快再改 30
  max_epoch: 20

  log_freq: 20
  accum_grad_iters: 1
  grad_norm_clip: null

  output_dir: "output/agiqa3k/quick"

  evaluate: False
  eval_freq: 1
  save_freq: 5

  # Trainer 原生支持 GradScaler
  amp: True

  device: "cuda"
  distributed: False
  dist_url: "env://"
```

---

# 11. 为什么今晚不做 Image2Prompt 预训练

IP-IQA 论文完整方法还包含 Image2Prompt incremental pretraining。

今晚直接跳过：

```yaml
load_finetuned: False
```

只加载 OpenAI CLIP RN50 权重。

原因：

- 我们现在的目标是拿到一个 AGIQA-3K supervised preliminary result；
- 不需要为了明天面试先复刻其额外大规模预训练阶段；
- baseline 和 Ours 都使用同一个 initialization，即可公平验证 local branch 本身。

---

# 12. 今晚实验顺序（非常重要）

## E0：先跑原始 baseline

先不要改模型，跑：

```text
IP-IQA
CLIP RN50 initialization
single split
seed 42
20 epochs
```

记录：

```text
quality SRCC
quality PLCC
quality KRCC
alignment SRCC
alignment PLCC
training time
max GPU memory
```

这一步必须先有结果。

---

## E1：Full Ours

加入：

```text
projection
+
1 attention block
+
patch score / weight
+
gated quality fusion
```

其余全部和 E0 一样：

```text
same seed
same split
same epochs
same LR
same image size
same CLIP initialization
```

最重要的比较：

```text
E1 quality SRCC - E0 quality SRCC
E1 quality PLCC - E0 quality PLCC
```

这才是明天最公平的实验结果。

---

# 13. 如果今晚还有时间：最小消融

优先级：

```text
E0  IP-IQA baseline
E1  + Patch Score/Weight（不加 attention）
E2  + Attention + Patch Score/Weight（Full）
```

这样你可以得到：

| Model | Local Attn | Patch Weight | Quality SRCC | Quality PLCC |
|---|---:|---:|---:|---:|
| IP-IQA | × | × | X | X |
| + Weighted Local | × | ✓ | X | X |
| Full Ours | ✓ | ✓ | X | X |

如果只能跑两次：

```text
E0 baseline
E2 full
```

就够。

---

# 14. 可以放在 PPT 的已有基线参考

AGIQA-3K 文献中常见的 paper-reported perception-quality 数字包括：

| Method | SRCC | PLCC |
|---|---:|---:|
| CNNIQA | 0.7478 | 0.8469 |
| DBCNN | 0.8207 | 0.8759 |
| HyperIQA | 0.8355 | 0.8903 |
| CLIPIQA | 0.8426 | 0.8053 |
| IP-IQA | 0.8634 | 0.9116 |

**重要：这些是文献报告值，不能和你的单次 split 数字声称严格公平横比。**

PPT 最好分成：

### Published reference

```text
CNNIQA / DBCNN / HyperIQA / CLIPIQA / IP-IQA paper-reported
```

### Our single-split preliminary experiment

```text
IP-IQA reproduced      X / X
Ours                   Y / Y
```

面试重点讲：

```text
Y - X
```

而不是单纯说 `Y > 某篇 paper number`。

---

# 15. 成功标准

## 最低可展示

```text
能完整训练
SRCC/PLCC 正常
明显高于随机预测
能压过部分经典 paper-reported baseline（仅作为参考）
```

## 真正有意义

同一个 split 下：

```text
Ours >= reproduced IP-IQA
```

最好：

```text
Quality SRCC / PLCC 有小幅提升
Alignment 基本不下降
```

例如下面只是“理想结果形态示例”，不是预测：

```text
IP-IQA reproduced     SRCC 0.841
Ours                  SRCC 0.847
```

如果真实结果如此，明天就可以讲：

> “在相同单次随机划分下，增加局部失真感知后 quality SRCC 相比 reproduced IP-IQA 提升约 0.006；目前仍是初步实验，后续将补多次随机划分。”

---

# 16. 如果 OOM / 太慢怎么退

按顺序退：

### 方案 A

```yaml
batch_size: 8
amp: True
```

### 方案 B

```yaml
batch_size: 4
accum_grad_iters: 2
```

### 方案 C

把 local hidden_dim：

```text
256 -> 128
```

### 方案 D

今晚只跑：

```text
10 epochs smoke test
```

确认曲线正常后，再跑 20 epoch。

### 不建议今晚做

- 第二个视觉 backbone；
- 完整 AM-BQA；
- Image2Prompt 56 万图预训练；
- 10 次 CV；
- 100 epoch；
- 大范围超参搜索。

---

# 17. 如果 Full Ours 掉点：快速回退策略

不要立刻推翻 idea。

## 回退 1：只保留 Patch Weight

```text
projection
↓
patch score + patch weight
↓
q_local
```

删 attention。

如果这个涨，说明 attention 不适配，但 local weighted prediction 有价值。

## 回退 2：减小 gate

```python
local_gate_logit = -3.0
```

这样初始 local 权重约 4.7%。

## 回退 3：hidden_dim 降到 128

小数据集上可能更稳，也更快。

---

# 18. 建议代码文件结构

```text
IP-IQA/
├── ipiqa/
│   └── models/
│       ├── ipiqa.py
│       ├── utils.py
│       └── local_distortion.py      # 新增
│
├── ipiqa/projects/agiqa3k/
│   ├── ipiqa.yaml
│   └── ipiqa_quick.yaml             # 新增
│
├── cache/
│   ├── ckpt/clip/openai/resnet/RN50.pt
│   └── data/aigc_qa_3k/
│       ├── AGIQA-3K/
│       ├── data.csv
│       └── mos_joint.xlsx
│
└── train_agiqa3k.py
```

---

# 19. 明天面试时的方法描述（简洁版）

> “我的 baseline 是 IP-IQA，它使用 CLIP 提取图像和文本特征，并通过 cross-attention 建模生成图像和原 Prompt 的关系。但我认为生成图像中的手部、纹理、文字等局部伪影可能在全局语义融合中被弱化。因此我参考 AM-BQA 的 attention-based local modeling 和 weighted patch quality prediction，在 IP-IQA 已有 CLIP spatial feature 上增加轻量局部失真分支，并通过 learnable gate 与原 quality prediction 融合。由于复用了同一视觉 backbone，新增计算量主要来自一个 1×1 projection 和一个轻量 attention/head。”

实验描述：

> “目前为了快速验证，使用 AGIQA-3K 固定 seed 的一次 80/20 随机 hold-out，baseline 与改进方法使用完全相同的数据划分和训练配置；现在报告的是 preliminary result，后续会补多次随机划分验证稳定性。”

---

# 20. 今晚 Checklist

```text
[ ] clone IP-IQA
[ ] pip install -e .
[ ] 下载 RN50.pt
[ ] 下载 AGIQA-3K images
[ ] 获取 data.csv
[ ] 生成 mos_joint.xlsx
[ ] 配置路径
[ ] 先用 1 batch 跑 forward
[ ] baseline 跑 1 epoch smoke test
[ ] baseline 正式跑 20 epoch
[ ] 记录 SRCC / PLCC
[ ] 新建 local_distortion.py
[ ] 修改 ipiqa.py
[ ] Ours 跑 1 epoch smoke test
[ ] Ours 正式跑同样 20 epoch
[ ] 做 baseline vs Ours 表
[ ] 截 training log / GPU / result，明天可放 PPT
```

---

# 21. 最后一句原则

今晚不要追“论文最终版”。

今晚真正需要拿到的是：

```text
同一个 AGIQA-3K split
同一个 IP-IQA baseline
          ↓
Baseline X
vs
Ours Y
```

只要这个比较真实、可复现、解释清楚，就已经比只有论文 idea 强很多。

