"""Multi-Scale Dual-Attention local quality branch for IP-IQA.

Inspired by MS-SCANet (ICASSP 2025, github.com/mithila442/MS-SCANet): we adapt
its multi-scale dual-attention idea (channel attention + spatial attention) to
IP-IQA's CLIP spatial feature, without adding a second backbone or copying the
full MS-SCANet network.  Cross-Branch Attention 已移除（见 完整消融方案.md）。

Structure (from ``feat = [B, 2048, H, W]``):
    f0 = 1x1 conv (2048 -> dim)
    fine   = f0                      (H x W tokens)
    coarse = avg_pool2d(f0, 2)       (H/2 x W/2 tokens)
    (可选) fine/coarse -> ChannelBlock -> SpatialBlock   [use_dual_attention]
    -> coarse upsample -> concat -> 1x1 fuse
    -> residual-gated refinement: f0 + sigmoid(gate) * delta
    -> patch score (+ weight) -> mean / weighted-average -> q_local

Only depends on ``torch``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """Simple MLP with GELU activation and dropout (like a ViT MLP)."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class ChannelAttention(nn.Module):
    """Sequence-mean channel recalibration (SE) on token sequences [B, N, D].

    NOTE: an adaptation of MS-SCANet's ChannelAttention (which squeezes each
    patch with a 1x1 conv over [B*N, D, 1, 1]). Here we use a global
    sequence-mean channel gate instead, cheaper and stable on CLIP features.
    """

    def __init__(self, dim=256, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 8)),
            nn.GELU(),
            nn.Linear(max(dim // reduction, 8), dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, N, D]
        w = self.fc(x.mean(dim=1))           # [B, D]
        return x * w.unsqueeze(1)


class ChannelBlock(nn.Module):
    """norm -> channel attention (residual) -> norm -> MLP (residual)."""

    def __init__(self, dim=256, mlp_ratio=2.0, drop=0.1, reduction=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.channel_attn = ChannelAttention(dim, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # x: [B, N, D]
        x = x + self.channel_attn(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialBlock(nn.Module):
    """norm -> global self-attention (residual) -> norm -> MLP (residual).

    MS-SCANet uses window attention here; on CLIP's 16x16 / 8x8 grids the
    token count is small, so we use plain global self-attention instead.
    """

    def __init__(self, dim=256, num_heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # x: [B, N, D]
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MSDualAttentionRefiner(nn.Module):
    """Multi-scale dual-attention feature refiner on CLIP spatial features.

    NOTE: Cross-Branch Attention 已移除（见 完整消融方案.md）。
    fine / coarse 各自独立做 channel + spatial attention 后，直接
    resize + concat + 1x1 fuse 完成多尺度融合。
    """

    def __init__(self, in_channels=2048, dim=256, num_heads=4, mlp_ratio=2.0,
                 drop=0.1, refine_gate_init=-2.0, use_dual_attention=True):
        super().__init__()
        self.dim = dim
        self.use_dual_attention = use_dual_attention

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.GELU(),
        )

        if use_dual_attention:
            self.fine_channel = ChannelBlock(dim, mlp_ratio, drop)
            self.coarse_channel = ChannelBlock(dim, mlp_ratio, drop)

            self.fine_spatial = SpatialBlock(dim, num_heads, mlp_ratio, drop)
            self.coarse_spatial = SpatialBlock(dim, num_heads, mlp_ratio, drop)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
        )

        # 一开始主要信任原 E1 feature（sigmoid(-2) ≈ 0.119）
        self.refine_gate_logit = nn.Parameter(torch.tensor(float(refine_gate_init)))

    def forward(self, feat):
        # feat: [B, in_channels, H, W]  (H=W=16 at 512 input)
        f0 = self.proj(feat)                          # [B, dim, H, W]
        B, C, H, W = f0.shape

        if self.use_dual_attention:
            fine_map = f0
            coarse_map = F.avg_pool2d(f0, kernel_size=2)  # [B, dim, H//2, W//2]

            fine = fine_map.flatten(2).transpose(1, 2)    # [B, H*W, dim]
            coarse = coarse_map.flatten(2).transpose(1, 2)  # [B, H/2*W/2, dim]

            fine = self.fine_channel(fine)
            fine = self.fine_spatial(fine)
            coarse = self.coarse_channel(coarse)
            coarse = self.coarse_spatial(coarse)

            fine_map = fine.transpose(1, 2).reshape(B, C, H, W)
            coarse_map = coarse.transpose(1, 2).reshape(B, C, H // 2, W // 2)
        else:
            # A1：仅多尺度，无 dual attention
            fine_map = f0
            coarse_map = F.avg_pool2d(f0, kernel_size=2)

        coarse_up = F.interpolate(
            coarse_map, size=(H, W), mode="bilinear", align_corners=False
        )

        delta = self.fuse(torch.cat([fine_map, coarse_up], dim=1))

        r = torch.sigmoid(self.refine_gate_logit)
        return f0 + r * delta


class MSLocalQualityBranch(nn.Module):
    """Multi-scale dual-attention refiner + patch score/(weight) head.

    Args:
        aggregation: "weighted" -> importance-aware weighted average (创新点3),
                     "mean"     -> plain mean of patch scores (A1/A2 用).
    """

    def __init__(self, in_channels=2048, dim=256, num_heads=4, mlp_ratio=2.0,
                 drop=0.1, refine_gate_init=-2.0, use_dual_attention=True,
                 aggregation="weighted"):
        super().__init__()
        self.aggregation = aggregation

        self.refiner = MSDualAttentionRefiner(
            in_channels=in_channels,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            refine_gate_init=refine_gate_init,
            use_dual_attention=use_dual_attention,
        )

        self.score_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1),
        )

        if aggregation == "weighted":
            self.weight_head = nn.Sequential(
                nn.Linear(dim, 128),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
        else:
            self.weight_head = None

    def forward(self, feat):
        # feat: [B, 2048, H, W]
        x = self.refiner(feat)                    # [B, dim, H, W]
        x = x.flatten(2).transpose(1, 2)          # [B, N, dim]

        patch_score = self.score_head(x)          # [B, N, 1]

        if self.aggregation == "weighted":
            patch_weight = self.weight_head(x)    # [B, N, 1]
            q_local = (patch_score * patch_weight).sum(dim=1) / \
                      (patch_weight.sum(dim=1) + 1e-6)       # [B, 1]
        else:  # "mean"
            q_local = patch_score.mean(dim=1)     # [B, 1]

        return q_local
