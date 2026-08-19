# local_branch —— IP-IQA 缝合模块（Local Distortion Branch）

完全自包含的 PyTorch 模块，**只依赖 `torch`**，不 import IP-IQA / AM-BQA 源码树。
可直接整包拷到服务器，接进 IP-IQA 主工程。

## 目录

```
local_branch/
├── __init__.py          # 包导出
├── local_distortion.py  # TransposeAttentionBlock + LocalDistortionBranch
├── fusion.py            # GatedLocalFusion（缝合 wrapper）
├── test_forward.py      # 独立 smoke test（随机张量，无需数据）
└── README.md
```

## 设计（对应 02_implementation_plan.md 第 2、3 节）

```text
IP-IQA CLIP spatial feature F: [B, 2048, 16, 16]   (input_resolution=512, RN50 stride=32)
        ↓
LocalDistortionBranch
  1×1 Conv 2048 -> 256
  transpose-attention block (可开关，做 E1/E2 消融)
  patch score_i + patch weight_i -> weighted avg -> q_local  [B, 1]

GatedLocalFusion:
  q_final = (1 - g) * q_base + g * q_local,   g = sigmoid(gate_logit)
  仅融合 quality 通道，alignment 通道原样透传。
```

- `spatial_dim` = `(input_resolution // 32) ** 2`。512 → 256；384 → 144。
- 初始 `gate_logit = -2.0` → `g ≈ 0.119`，模型先信原 IP-IQA 路径，再让 gate 自学。

## 独立测试

```bash
python test_forward.py          # CPU
python test_forward.py --cuda   # GPU
```

## 接入 IP-IQA（最小改动）

把本文件夹拷到服务器上（示例放 `IP-IQA/ipiqa/models/local_branch/`），然后改
`ipiqa/models/ipiqa.py`：

```python
from ipiqa.models.local_branch import GatedLocalFusion

# __init__ 中加入（spatial_dim 按实际 input_resolution 算）：
self.local_fusion = GatedLocalFusion(
    in_channels=2048,
    hidden_dim=256,
    spatial_dim=(input_resolution // 32) ** 2,
    gate_init=-2.0,
    use_attention=True,   # False 即 E1 消融（去掉 attention）
)

# forward 中把原来的：
#     return self.head(global_feat)
# 替换为：
base_output = self.head(global_feat)          # [B, 2]
return self.local_fusion(base_output, feat)   # [B, 2]，q 为融合值，alignment 不变
```

注意：`forward` 里需要能拿到 CLIP RN50 的 `feat`（IP-IQA 里 `feat = self.resnet50(x)`），
把 `self.local_fusion` 的调用放在 `global_feat` 计算之后即可。

## 消融对照（02_implementation_plan.md 第 13 节）

| Model | Local Attn | Patch Weight | 开关 |
|---|---|---:|---|
| IP-IQA baseline | × | × | 不接入本模块 |
| + Weighted Local (E1) | × | ✓ | `use_attention=False` |
| Full Ours (E2) | ✓ | ✓ | `use_attention=True` |
