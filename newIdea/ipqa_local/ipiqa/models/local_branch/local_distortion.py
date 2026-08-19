"""Local Distortion Branch for IP-IQA.

Self-contained PyTorch modules implementing the "缝合模块" (stitching module):

  - ``TransposeAttentionBlock``: lightweight transpose-attention, inspired by
    the AM-BQA ``TABlock`` idea, adapted to IP-IQA's CLIP spatial feature.
  - ``LocalDistortionBranch``: projection -> attention -> patch score/weight
    -> weighted-average ``q_local``.

Only depends on ``torch``. No imports from IP-IQA or AM-BQA source trees.
"""

import torch
import torch.nn as nn


class TransposeAttentionBlock(nn.Module):
    """Lightweight transpose-attention block (adapted from AM-BQA TABlock idea).

    Operates on channel-major tensors ``[B, C, N]`` where ``N`` is the flattened
    spatial dimension (patch count). Attention is computed across the channel
    dimension (transpose attention), with the spatial positions acting as the
    token sequence. This is a small, residual attention block -- NOT a full
    ViT / Swin / ART backbone.
    """

    def __init__(self, spatial_dim=256, drop=0.1):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.q = nn.Linear(spatial_dim, spatial_dim)
        self.k = nn.Linear(spatial_dim, spatial_dim)
        self.v = nn.Linear(spatial_dim, spatial_dim)
        self.scale = spatial_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, C, N]
        if x.size(-1) != self.spatial_dim:
            raise ValueError(
                f"TransposeAttentionBlock expects last dim == {self.spatial_dim}, "
                f"got {x.size(-1)}. Check spatial_dim = (input_resolution // 32)^2."
            )
        residual = x
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)
        out = self.drop(out)
        return out + residual


class LocalDistortionBranch(nn.Module):
    """Predict a local perceptual-quality score from IP-IQA's spatial feature.

    Args:
        in_channels: channel dim of the input spatial feature (2048 for CLIP RN50).
        hidden_dim:  channel dim after the 1x1 projection (recommended 256).
        spatial_dim: flattened spatial size N = H*W. For input_resolution 512 and
                     CLIP RN50 stride 32: N = (512//32)^2 = 256.
        drop:        dropout probability inside the MLP heads.
        use_attention: if True keep the transpose-attention block (E2 / Full).
                       if False, skip it -- patch score/weight only (E1 ablation).
    """

    def __init__(self, in_channels=2048, hidden_dim=256, spatial_dim=256,
                 drop=0.1, use_attention=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.use_attention = use_attention

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        self.attn = (
            TransposeAttentionBlock(spatial_dim=spatial_dim, drop=drop)
            if use_attention
            else nn.Identity()
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
        # feat: [B, in_channels, H, W]
        x = self.proj(feat)                # [B, hidden_dim, H, W]
        B, C, H, W = x.shape
        N = H * W
        if N != self.spatial_dim:
            raise ValueError(
                f"LocalDistortionBranch expects N = {self.spatial_dim}, got {N}. "
                f"Set spatial_dim = H*W of the projected feature."
            )

        x = x.flatten(2)                   # [B, C, N]
        x = self.attn(x)                   # [B, C, N]
        x = x.transpose(1, 2).contiguous() # [B, N, C]

        patch_score = self.score_head(x)   # [B, N, 1]
        patch_weight = self.weight_head(x) # [B, N, 1]

        q_local = (patch_score * patch_weight).sum(dim=1) / \
                  (patch_weight.sum(dim=1) + 1e-6)        # [B, 1]

        return q_local
