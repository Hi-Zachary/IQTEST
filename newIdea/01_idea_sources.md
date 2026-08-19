# Idea 来源与代码资源：IP-IQA + AM-BQA 局部失真感知

> 当前拟定的大创/面试方案：以 **IP-IQA** 为多模态 AGIQA 主干，保留其图像–Prompt 融合能力；从 **AM-BQA** 中抽取“注意力增强局部特征 + patch 质量/权重预测”的思想，增加一个轻量局部失真分支，只增强 **perceptual quality** 预测，不额外增加第二个视觉 backbone。

---

## 1. 主干论文：IP-IQA

### 论文信息

- **题目**：Bringing Textual Prompt to AI-Generated Image Quality Assessment
- **作者**：Bowen Qu, Haohui Li, Wei Gao
- **会议**：IEEE ICME 2024
- **DOI**：10.1109/ICME57554.2024.10688254
- **论文（arXiv）**：https://arxiv.org/abs/2403.18714
- **DOI 页面**：https://doi.org/10.1109/ICME57554.2024.10688254

### 官方源码

- **GitHub**：https://github.com/Coobiw/IP-IQA
- 克隆：

```bash
git clone https://github.com/Coobiw/IP-IQA.git
cd IP-IQA
```

### 我们从 IP-IQA 保留什么

IP-IQA 是我们的 **baseline / 主工程代码**。核心保留：

1. OpenAI CLIP RN50 视觉编码器；
2. CLIP 文本编码器；
3. Prompt 与图像空间特征的 cross-attention / TextAttentionPool2d；
4. global visual feature + prompt-conditioned visual feature 的融合；
5. AGIQA-3K 的 quality + alignment 双输出训练框架；
6. 原仓库的数据读取、指标计算、训练器和单次随机划分代码。

### 最值得直接阅读的源码

#### 主模型
https://github.com/Coobiw/IP-IQA/blob/master/ipiqa/models/ipiqa.py

关注：

```python
txt_feat = self.encode_text(text)
feat = self.resnet50(x)
global_visual = self.attnpool(feat)
global_txt = self.txt_attnpool(feat, txt_feat)
global_feat = torch.cat([global_visual, global_txt], dim=-1)
return self.head(global_feat)
```

这里的 `feat` 就是我们增加局部失真分支最自然的接口。

#### Cross-Attention 实现
https://github.com/Coobiw/IP-IQA/blob/master/ipiqa/models/utils.py

重点类：

- `MultiHeadCrossAttention`
- `TextAttentionPool2d`

#### AGIQA-3K 训练入口
https://github.com/Coobiw/IP-IQA/blob/master/train_agiqa3k.py

这个文件已经自带随机 80/20 划分逻辑，并支持：

```bash
--num_cv 1
```

因此今晚不需要自己重写 K-fold。

#### AGIQA-3K 双分数任务
https://github.com/Coobiw/IP-IQA/blob/master/ipiqa/tasks/agiqa_doublescore.py

它把输出解释为：

```text
output[:, 0] -> perceptual quality
output[:, 1] -> image-text alignment
```

我们的局部分支只修改第 0 个 quality 输出，第 1 个 alignment 尽量保持原 IP-IQA 路径。

#### 官方 AGIQA-3K 配置
https://github.com/Coobiw/IP-IQA/blob/master/ipiqa/projects/agiqa3k/ipiqa.yaml

原配置大致是：CLIP RN50、512×512、batch 32、100 epoch、lr 1e-5。今晚为了速度会采用更短的 quick 配置，详见第二个文档。

---

## 2. 被缝合模块来源：AM-BQA

### 论文信息

- **题目**：AM-BQA: Enhancing blind image quality assessment using attention retractable features and multi-dimensional learning
- **期刊**：Image and Vision Computing
- **卷期**：Volume 147, July 2024, 105076
- **DOI**：10.1016/j.imavis.2024.105076
- **论文页面**：https://doi.org/10.1016/j.imavis.2024.105076

### 官方源码

- **GitHub**：https://github.com/adhikariastha5/AM-BQA
- 克隆：

```bash
git clone https://github.com/adhikariastha5/AM-BQA.git
cd AM-BQA
```

### AM-BQA 完整方法包含什么

AM-BQA 完整模型比我们需要的更重，包含：

- ViT 特征提取；
- transpose / dual-key attention；
- attention retractable transformer；
- 多阶段特征增强；
- patch-level score 与 patch-level weight 加权预测。

我们 **不会把整个 AM-BQA 搬进 IP-IQA**，否则会多一个完整视觉网络，训练速度和代码复杂度都会变差。

### 我们真正借用的两个思想

#### A. 注意力增强的局部/patch 特征交互

可参考：

https://github.com/adhikariastha5/AM-BQA/blob/main/models/ambqa_withart.py

其中可以看到 `TABlock`、`MultiHeadDualBlock` 等 attention 结构。

我们只抽象成一个轻量版本：

```text
IP-IQA spatial feature
        ↓
1×1 projection
        ↓
1 个 transpose-attention-style block
        ↓
enhanced local feature
```

不会复制 AM-BQA 的 ViT、Swin/ART 等完整 backbone。

#### B. Patch Score + Patch Weight

最直观的参考文件：

https://github.com/adhikariastha5/AM-BQA/blob/main/models/ambqa.py

AM-BQA 对每个局部 patch 分别预测：

```text
quality score_i
importance weight_i
```

再进行：

```text
sum(score_i * weight_i) / sum(weight_i)
```

我们保留这个思想，得到一个 `q_local`，让局部异常区域对最终质量预测产生更直接的影响。

---

## 3. 我们的缝合点为什么自然

IP-IQA 已经在 CLIP RN50 中得到空间 feature map：

```text
Image
  ↓
CLIP RN50
  ↓
spatial feature F
  ├── 原 IP-IQA：global pooling + Prompt-conditioned attention
  │
  └── 新分支：Local Distortion Branch
```

因此不需要再次对图片跑一个新的 backbone。

拟采用：

```text
F
├── Original IP-IQA Branch -> q_base / alignment
│
└── Local Distortion Branch
     -> projection
     -> lightweight attention
     -> patch score + patch weight
     -> q_local

q_base + q_local
       ↓
learnable gated fusion
       ↓
q_final
```

核心研究假设：

> IP-IQA 主要利用 Prompt 和视觉语义信息帮助 AGIQA；但 AIGC 图像的手指、眼睛、纹理、文字、边缘等局部生成异常不一定能被全局/语义融合充分保留。因此增加一个显式的局部质量感知分支，使模型同时利用 cross-modal semantic evidence 和 local distortion evidence。

这属于 **我们提出并需要实验验证的融合假设**，不是两篇原论文已经证明过的结论。

---

## 4. 数据集来源：AGIQA-3K

### 数据集论文

- **题目**：AGIQA-3K: An Open Database for AI-Generated Image Quality Assessment
- **期刊**：IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2023
- **DOI**：10.1109/TCSVT.2023.3319020
- **论文**：https://arxiv.org/abs/2306.04717
- **DOI**：https://doi.org/10.1109/TCSVT.2023.3319020

### 官方数据集仓库

https://github.com/lcysyzxdxc/AGIQA-3k-Database

仓库中的 `data.csv` 已经包含：

- image name
- generation prompt
- perception MOS
- perception STD
- alignment MOS
- alignment STD

### 图片直接下载

官方 README 当前提供：

- Google Drive：
  https://drive.google.com/file/d/1ObuOZ6YZqZuxe4oRlaf3kdOBlTRg2GE4/view?usp=sharing
- 夸克：
  https://pan.quark.cn/s/10187e65d5c1

下载后解压 `AGIQA-3K.zip`。

---

## 5. CLIP RN50 权重

IP-IQA 官方仓库使用 OpenAI CLIP RN50。

官方权重直链：

https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt

推荐放到：

```text
IP-IQA/cache/ckpt/clip/openai/resnet/RN50.pt
```

---

## 6. 最终引用关系

如果之后写大创书/答辩，可以这样描述来源：

```text
主干：IP-IQA (ICME 2024)
  └─ 图像–Prompt 多模态质量评价框架
     └─ CLIP visual/text representation
     └─ image-prompt fusion / cross attention

局部模块思想：AM-BQA (Image and Vision Computing 2024)
  └─ attention-enhanced local feature modeling
  └─ patch quality score + adaptive patch weighting

我们的工作
  └─ 不增加第二个 backbone
  └─ 在 IP-IQA 的 CLIP spatial feature 上增加轻量 local distortion branch
  └─ 只增强 perceptual quality 分支
  └─ 与原 cross-modal quality prediction 做 gated fusion
```

---

## 7. 今晚最重要的代码入口（按阅读顺序）

1. IP-IQA 主模型：
   https://github.com/Coobiw/IP-IQA/blob/master/ipiqa/models/ipiqa.py
2. IP-IQA cross-attention：
   https://github.com/Coobiw/IP-IQA/blob/master/ipiqa/models/utils.py
3. IP-IQA 数据划分：
   https://github.com/Coobiw/IP-IQA/blob/master/train_agiqa3k.py
4. IP-IQA AGIQA-3K Dataset：
   https://github.com/Coobiw/IP-IQA/blob/master/ipiqa/datasets/agiqa_datasets.py
5. AM-BQA 简洁 weighted patch head：
   https://github.com/adhikariastha5/AM-BQA/blob/main/models/ambqa.py
6. AM-BQA 更完整 attention 实现：
   https://github.com/adhikariastha5/AM-BQA/blob/main/models/ambqa_withart.py
7. AGIQA-3K 数据：
   https://github.com/lcysyzxdxc/AGIQA-3k-Database

