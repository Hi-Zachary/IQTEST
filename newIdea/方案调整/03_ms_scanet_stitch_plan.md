# 新缝合方案：IP-IQA + MS-SCANet 多尺度双注意力局部质量分支

> 当前工程：`newIdea/ipqa_local/`
>
> 当前已验证有效：IP-IQA + Weighted Local
>
> 新目标：保留当前已经有效的 patch score / patch weight 预测头，在其前面加入一个真正具有网络结构厚度的多尺度双注意力特征增强模块。

## 1. 当前仓库现状

当前核心文件：

```text
newIdea/ipqa_local/ipiqa/models/ipiqa.py
newIdea/ipqa_local/ipiqa/models/local_branch/local_distortion.py
newIdea/ipqa_local/ipiqa/models/local_branch/fusion.py
newIdea/ipqa_local/ipiqa/projects/agiqa3k/ipiqa_quick.yaml
newIdea/ipqa_local/ipiqa/projects/agiqa3k/ipiqa_ours_noattn.yaml
newIdea/ipqa_local/splits/seed42.json
newIdea/RESULTS.md
```

当前 IP-IQA 前向中已经有：

```python
feat = self.resnet50(x)
```

512 输入时，当前工程的 `feat` 是：

```text
[B, 2048, 16, 16]
```

原 IP-IQA 继续从这里做：

```text
feat
├─ attnpool -> global_visual
└─ txt_attnpool(feat, txt_feat) -> global_txt

concat(global_visual, global_txt)
↓
head
↓
base_output = [quality, alignment]
```

当前已接入的 E1 局部分支：

```text
feat
↓
1×1 Conv 2048 -> 256
↓
patch score + patch weight
↓
weighted average
↓
q_local
↓
与 q_base gated fusion
```

同一 seed42 / 同一 80:20 split 下：

```text
IP-IQA baseline qual SROCC = 0.8153
Weighted Local qual SROCC  = 0.8262
```

因此新的方案不应该推倒现有工程，而应当：

> **保留当前 Weighted Local 作为稳定预测头，只替换它前面的局部特征处理部分。**

---

## 2. 新的主要参考论文：MS-SCANet

论文：

**MS-SCANet: A Multiscale Transformer-Based Architecture with Dual Attention for No-Reference Image Quality Assessment**

正式发表：

- IEEE ICASSP 2025
- Authors: Mayesha Maliha R. Mithila, Mylène C. Q. Farias
- Affiliation: Department of Computer Science, Texas State University
- DOI: https://doi.org/10.1109/ICASSP49660.2025.10887759
- arXiv 后补版本: https://arxiv.org/abs/2602.04032

源码：

- GitHub: https://github.com/mithila442/MS-SCANet
- Core model: https://github.com/mithila442/MS-SCANet/blob/main/ms_scanet.py

仓库有：

```text
ms_scanet.py
train_pre.py
train_final.py
data_loader.py
config.py
utils.py
```

它不是空仓库。

### 我们真正借的模块

MS-SCANet 的核心可以拆成：

```text
Multi-scale dual branches
+
Channel Attention
+
Spatial Attention
+
Cross-Branch Attention
```

源码中直接对应：

```python
ChannelBlock
SpatialBlock
CrossBranchAttention
MultiScaleDualAttentionTransformer
```

我们借：

1. 双尺度思想
2. ChannelBlock
3. SpatialBlock
4. CrossBranchAttention

我们不搬：

```text
原 PatchEmbed
原 RGB 双尺度输入
原 prediction head
原完整训练流程
原 consistency losses
```

因为 IP-IQA 已经有 CLIP RN50，再跑完整 MS-SCANet 不划算。

---

## 3. AM-BQA 在新方案里的定位

AM-BQA 不再承担“第二大网络模块”的角色。

论文：

**AM-BQA: Enhancing blind image quality assessment using attention retractable features and multi-dimensional learning**

- Image and Vision Computing, 2024
- Volume 147, Article 105076
- DOI: https://doi.org/10.1016/j.imavis.2024.105076
- Code: https://github.com/adhikariastha5/AM-BQA

现在只保留它对当前项目最有价值、且已经被现有 E1 结果支持的思想：

```text
patch-wise quality score
+
patch-wise importance weight
+
weighted aggregation
```

所以最终来源关系：

```text
IP-IQA
→ 完整跨模态主框架

MS-SCANet
→ 多尺度局部特征
→ channel attention
→ spatial attention
→ cross-scale interaction

AM-BQA
→ patch score / patch weight 聚合
```

---

## 4. 新完整网络结构

```text
                           Image + Prompt
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ↓                             ↓
             CLIP RN50                     CLIP Text
                  │                             │
                  ↓                             │
        F = [B,2048,16,16]                      │
                  │                             │
        ┌─────────┴────────────┐                │
        │                      │                │
        ↓                      ↓                │
 Original IP-IQA      Multi-Scale Local Branch  │
        │                      │                │
        │              1×1 Conv 2048→256       │
        │                      │                │
        │                 F0 16×16              │
        │                 /      \             │
        │                /        \            │
        │               ↓          ↓            │
        │          Fine Branch   Coarse Branch  │
        │            16×16         8×8           │
        │               │          │            │
        │         Channel Attn  Channel Attn     │
        │               │          │            │
        │         Spatial Attn  Spatial Attn     │
        │               │          │            │
        │               └────┬─────┘            │
        │                    ↓                  │
        │          Cross-Branch Attention       │
        │            Fine ↔ Coarse              │
        │                    │                  │
        │         coarse upsample → 16×16       │
        │                    │                  │
        │           feature concatenation       │
        │                    │                  │
        │              1×1 Conv fusion          │
        │                    │                  │
        │            residual refinement        │
        │                    │                  │
        │             refined feature           │
        │                    │                  │
        │           Patch Score Head            │
        │                  +                    │
        │           Patch Weight Head           │
        │                    │                  │
        │                 q_local               │
        │                    │                  │
        ↓                    │                  │
     q_base ─────────────────┘                  │
        │                                       │
        ↓                                       │
     Gated Fusion                               │
        ↓                                       │
   Final Quality                                │
                                                │
 Alignment 保持原 IP-IQA 路径 ──────────────────┘
```

一句话：

> IP-IQA 负责全局视觉与 prompt 跨模态建模；新增分支从 CLIP 空间特征构建 fine/coarse 两个尺度，通过通道、空间和跨尺度交互增强局部质量表示，再用 patch-wise score/weight 输出局部质量证据。

---

## 5. 为什么这个接口自然

当前 `feat` 本身就是空间特征图：

```text
[B,2048,16,16]
```

所以不用再：

```text
RGB → 第二个 ViT / Swin / ResNet
```

而是：

```text
feat
↓
1×1 Conv
↓
[B,256,16,16]
```

构造两个尺度：

```python
fine = f0
coarse = F.avg_pool2d(f0, 2)
```

得到：

```text
fine:
[B,256,16,16]
256 tokens

coarse:
[B,256,8,8]
64 tokens
```

没有第二次 CLIP forward。

---

## 6. 新增文件

建议新建：

```text
newIdea/ipqa_local/ipiqa/models/local_branch/ms_dual_local.py
```

里面放：

```python
ChannelAttention
ChannelBlock
SpatialBlock
CrossBranchAttention
MSDualAttentionRefiner
MSLocalQualityBranch
```

不要复制整个 `ms_scanet.py`。

---

## 7. Feature Projection

继续复用当前 E1 的接口：

```python
self.proj = nn.Sequential(
    nn.Conv2d(2048, 256, kernel_size=1),
    nn.GELU(),
)
```

得到：

```text
f0 = [B,256,16,16]
```

第一轮不要改 hidden_dim，保持 256，便于和 E1 公平比较。

---

## 8. Multi-Scale

```python
fine_map = f0
coarse_map = F.avg_pool2d(f0, kernel_size=2)
```

转 token：

```python
fine = fine_map.flatten(2).transpose(1, 2)
# [B,256,256]

coarse = coarse_map.flatten(2).transpose(1, 2)
# [B,64,256]
```

---

## 9. Channel Attention

建议做成轻量 residual channel refinement：

```python
class ChannelAttention(nn.Module):
    def __init__(self, dim=256, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x.mean(dim=1))
        return x * w.unsqueeze(1)
```

使用时：

```python
x = x + self.channel_attn(self.norm(x))
```

fine / coarse 各一套。

推荐：

```text
dim = 256
reduction = 16
```

---

## 10. Spatial Attention

不要再复用已经掉点的 AM-BQA transpose attention。

换标准 token self-attention：

```python
class SpatialBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, mlp_ratio=2.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
```

推荐第一版：

```text
num_heads = 4
mlp_ratio = 2
dropout = 0.1
```

不要一上来设得很大。

---

## 11. Cross-Branch Attention

这是新模块里最值得保留的部分。

目标：

```text
Fine 看 Coarse
Coarse 也看 Fine
```

输入：

```text
Fine   [B,256,256]
Coarse [B, 64,256]
```

实现可以写成：

```python
class CrossBranchAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()

        self.norm_f = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)

        self.f_to_c = nn.MultiheadAttention(
            dim, heads, batch_first=True
        )

        self.c_to_f = nn.MultiheadAttention(
            dim, heads, batch_first=True
        )

    def forward(self, fine, coarse):

        f = self.norm_f(fine)
        c = self.norm_c(coarse)

        fine_delta, _ = self.f_to_c(
            query=f,
            key=c,
            value=c,
        )

        coarse_delta, _ = self.c_to_f(
            query=c,
            key=f,
            value=f,
        )

        fine = fine + fine_delta
        coarse = coarse + coarse_delta

        return fine, coarse
```

这比“两个尺度各自算完再 concat”更有结构意义。

---

## 12. 跨尺度 Fusion

恢复空间：

```python
fine_map = fine.transpose(1,2).reshape(B,256,16,16)

coarse_map = coarse.transpose(1,2).reshape(B,256,8,8)
```

coarse 上采样：

```python
coarse_up = F.interpolate(
    coarse_map,
    size=(16,16),
    mode="bilinear",
    align_corners=False,
)
```

拼接：

```python
fusion_input = torch.cat(
    [fine_map, coarse_up],
    dim=1
)
```

shape：

```text
[B,512,16,16]
```

融合：

```python
self.scale_fusion = nn.Sequential(
    nn.Conv2d(512,256,1),
    nn.GELU(),
)
```

---

## 13. 建议加入小残差门

之前“直接加 attention”掉过点，所以这次不要强制替换原特征。

建议：

```python
r = torch.sigmoid(self.refine_gate_logit)

refined = f0 + r * fusion_feature
```

初始化：

```python
refine_gate_logit = -2.0
```

所以：

```text
r ≈ 0.119
```

一开始模型仍然主要依赖原 E1 feature。

这个 gate 是我们为了稳定迁移做的适配，不是 MS-SCANet 原论文公式。

---

## 14. 继续复用 Weighted Patch Prediction

当前 E1 已经有效，所以预测头先完全不改。

```python
x = refined.flatten(2).transpose(1,2)

patch_score = self.score_head(x)
patch_weight = self.weight_head(x)

q_local = (
    patch_score * patch_weight
).sum(dim=1) / (
    patch_weight.sum(dim=1) + 1e-6
)
```

所以区别是：

```text
旧 E1:
f0
↓
patch score / weight

新 E3:
f0
↓
multi-scale dual-attention refinement
↓
patch score / weight
```

---

## 15. Final Quality Fusion 第一轮不要改

继续用当前 `fusion.py`：

```python
g = torch.sigmoid(self.local_gate_logit)

q_final = (1-g) * q_base + g * q_local
```

alignment：

```python
align_final = align_base
```

第一轮只改局部 feature extractor。

不要同时改：

```text
新 feature
+
新 final fusion
+
新 loss
```

否则实验不好解释。

---

## 16. 推荐的 `ms_dual_local.py` 骨架

```python
class MSDualAttentionRefiner(nn.Module):

    def __init__(
        self,
        in_channels=2048,
        dim=256,
        num_heads=4,
        mlp_ratio=2.0,
        refine_gate_init=-2.0,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, 1),
            nn.GELU(),
        )

        self.fine_channel = ChannelBlock(dim)
        self.coarse_channel = ChannelBlock(dim)

        self.fine_spatial = SpatialBlock(
            dim, num_heads, mlp_ratio
        )

        self.coarse_spatial = SpatialBlock(
            dim, num_heads, mlp_ratio
        )

        self.cross = CrossBranchAttention(
            dim, num_heads
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
        )

        self.refine_gate_logit = nn.Parameter(
            torch.tensor(float(refine_gate_init))
        )

    def forward(self, feat):

        f0 = self.proj(feat)

        fine_map = f0
        coarse_map = F.avg_pool2d(f0, 2)

        fine = fine_map.flatten(2).transpose(1,2)
        coarse = coarse_map.flatten(2).transpose(1,2)

        fine = self.fine_channel(fine)
        fine = self.fine_spatial(fine)

        coarse = self.coarse_channel(coarse)
        coarse = self.coarse_spatial(coarse)

        fine, coarse = self.cross(
            fine, coarse
        )

        B = feat.size(0)

        fine_map = fine.transpose(1,2).reshape(
            B, 256, 16, 16
        )

        coarse_map = coarse.transpose(1,2).reshape(
            B, 256, 8, 8
        )

        coarse_up = F.interpolate(
            coarse_map,
            size=(16,16),
            mode="bilinear",
            align_corners=False,
        )

        delta = self.fuse(
            torch.cat(
                [fine_map, coarse_up],
                dim=1
            )
        )

        r = torch.sigmoid(
            self.refine_gate_logit
        )

        return f0 + r * delta
```

再封装：

```python
class MSLocalQualityBranch(nn.Module):

    def __init__(
        self,
        in_channels=2048,
        dim=256,
        num_heads=4,
        mlp_ratio=2.0,
        drop=0.1,
        refine_gate_init=-2.0,
    ):
        super().__init__()

        self.refiner = MSDualAttentionRefiner(
            in_channels=in_channels,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            refine_gate_init=refine_gate_init,
        )

        self.score_head = nn.Sequential(
            nn.Linear(dim,128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128,1),
        )

        self.weight_head = nn.Sequential(
            nn.Linear(dim,128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )

    def forward(self, feat):

        x = self.refiner(feat)
        x = x.flatten(2).transpose(1,2)

        score = self.score_head(x)
        weight = self.weight_head(x)

        return (
            score * weight
        ).sum(dim=1) / (
            weight.sum(dim=1) + 1e-6
        )
```

---

## 17. 修改当前 `fusion.py`

不要删旧的 E1 branch。

增加：

```python
branch_type
```

例如：

```python
if branch_type == "weighted":

    self.local_branch = LocalDistortionBranch(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        use_attention=False,
    )

elif branch_type == "msda":

    self.local_branch = MSLocalQualityBranch(
        in_channels=in_channels,
        dim=hidden_dim,
        num_heads=ms_num_heads,
        mlp_ratio=ms_mlp_ratio,
        refine_gate_init=ms_refine_gate_init,
    )
```

这样：

```text
branch_type="weighted"
```

永远能回到现在已经验证的 E1。

---

## 18. 修改当前 `ipiqa.py`

当前 forward 已经是非常好的缝合接口：

```python
base_output = self.head(global_feat)

if self.local_fusion is not None:
    return self.local_fusion(
        base_output,
        feat
    )

return base_output
```

这个 forward **不需要改**。

只在配置中增加：

```python
local_branch_type="weighted"
ms_num_heads=4
ms_mlp_ratio=2.0
ms_refine_gate_init=-2.0
```

然后创建 `GatedLocalFusion` 时传进去。

---

## 19. 新配置文件

新增：

```text
ipiqa/projects/agiqa3k/ipiqa_msda.yaml
```

以：

```text
ipiqa_ours_noattn.yaml
```

为基础。

model 部分：

```yaml
model:
  arch: "ipiqa"

  base_ckpt: ../data/ckpt/clip/openai/resnet/RN50.pt

  input_resolution: 512

  output_dim: 2

  load_finetuned: False

  head_scale: 10.

  use_mlp_head: True

  dropout_rate: 0.5

  freeze_text: True

  qa_token: True


  use_local_branch: True

  local_branch_type: "msda"

  local_hidden_dim: 256

  local_gate_init: -2.0


  ms_num_heads: 4

  ms_mlp_ratio: 2.0

  ms_refine_gate_init: -2.0
```

第一轮 run 部分完全复用 E1：

```yaml
batch_size: 8
batch_size_val: 16
max_epoch: 20
init_lr: 1e-5
amp: True
split_file: "splits/seed42.json"
distributed: False
```

不要同时换 epoch、lr、split、resolution。

---

## 20. 第一轮实验设计

已有：

```text
E0:
IP-IQA baseline

E1:
IP-IQA
+ Weighted Local
```

新增：

```text
E3:
IP-IQA
+ MSDA Local
+ Weighted Prediction
```

第一轮最重要：

```text
E3 vs E1
```

因为 E1 才是当前公平的 strongest local baseline。

---

## 21. 后续消融

如果 E3 有提升，再补：

| Experiment | Multi-scale | Channel/Spatial | Cross-scale | Patch Weight |
|---|---:|---:|---:|---:|
| IP-IQA | × | × | × | × |
| Weighted Local | × | × | × | ✓ |
| Dual Scale | ✓ | × | × | ✓ |
| Dual Attention | ✓ | ✓ | × | ✓ |
| Full | ✓ | ✓ | ✓ | ✓ |

这张表以后非常适合大创或论文的网络消融。

---

## 22. 第一版不要加 MS-SCANet 的额外 loss

MS-SCANet 还包含 consistency 类 loss。

第一版不要搬。

原因：

1. 需要改 trainer / task
2. 需要再调 loss weight
3. 网络和 loss 同时变化会难以定位贡献
4. 当前项目更想突出网络结构，而不是复杂损失

第一版只拿网络模块。

---

## 23. 算力判断

不会新增：

```text
第二个 CLIP
第二个 ViT backbone
第二次图像编码
```

新增主要是：

```text
Fine: 256 tokens × 256 dim
Coarse: 64 tokens × 256 dim

2 × channel block
2 × spatial block
1 × cross-branch attention
```

最重的仍然是：

```text
512×512 CLIP RN50
```

如果 OOM / 太慢，按顺序减：

```text
num_heads:
4 → 2

hidden_dim:
256 → 192 / 128

mlp_ratio:
2 → 1.5

batch:
8 → 4 + grad accumulation
```

第一优先级不要改 512 resolution，否则和 E0/E1 不再完全公平。

---

## 24. 最终方法描述

推荐中文：

**多尺度局部-跨模态协同的 AI 生成图像质量评价网络**

或者：

**基于多尺度局部质量增强的跨模态 AI 生成图像质量评价方法**

方法描述：

> 在 IP-IQA 跨模态质量建模框架基础上，引入多尺度局部质量感知分支。该分支从 CLIP 空间视觉特征构造细粒度和粗粒度表示，并通过通道注意力、空间注意力以及跨尺度交互联合增强局部失真特征；随后利用 patch-wise quality score 与 importance weight 聚合局部质量证据，并与原 IP-IQA 的全局跨模态质量预测进行自适应融合。

---

## 25. 参考链接汇总

### IP-IQA

- Paper: Bringing Textual Prompt to AI-Generated Image Quality Assessment
- ICME 2024
- DOI: https://doi.org/10.1109/ICME57554.2024.10688254
- arXiv: https://arxiv.org/abs/2403.18714
- Code: https://github.com/Coobiw/IP-IQA

### MS-SCANet

- Paper: MS-SCANet: A Multiscale Transformer-Based Architecture with Dual Attention for No-Reference Image Quality Assessment
- ICASSP 2025
- DOI: https://doi.org/10.1109/ICASSP49660.2025.10887759
- arXiv: https://arxiv.org/abs/2602.04032
- Code: https://github.com/mithila442/MS-SCANet
- Core model: https://github.com/mithila442/MS-SCANet/blob/main/ms_scanet.py

### AM-BQA

- Paper: AM-BQA: Enhancing blind image quality assessment using attention retractable features and multi-dimensional learning
- Image and Vision Computing, 2024
- DOI: https://doi.org/10.1016/j.imavis.2024.105076
- Code: https://github.com/adhikariastha5/AM-BQA

---

## 26. 最实际的开发顺序

```text
Step 1
不要改现有 E1

Step 2
新增 ms_dual_local.py

Step 3
随机 tensor smoke test

[B,2048,16,16]
→
[B,1]

Step 4
fusion.py 加 branch_type

Step 5
ipiqa.py 只加配置字段
forward 不改

Step 6
新增 ipiqa_msda.yaml

Step 7
同 seed42 / 同 20 epoch 跑一次 E3

Step 8
先比较 E3 vs E1

Step 9
如果上涨，再补消融
```

---

## 27. 一句话结论

当前最推荐的升级不是再给 IP-IQA 后面挂一个小 attention，而是：

```text
IP-IQA CLIP spatial feature
↓
fine / coarse 双尺度
↓
Channel Attention
↓
Spatial Attention
↓
Cross-Branch Attention
↓
Residual Multi-Scale Fusion
↓
Patch Score + Patch Weight
↓
q_local
↓
与 IP-IQA q_base Gated Fusion
```

即：

**IP-IQA + MS-SCANet-inspired Multi-Scale Dual-Attention Local Quality Branch + AM-BQA-style Patch-Weighted Prediction。**
