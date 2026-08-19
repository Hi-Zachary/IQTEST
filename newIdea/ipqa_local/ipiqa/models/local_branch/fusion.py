"""Gated fusion that stitches the Local Distortion Branch into IP-IQA.

The wrapper takes two precomputed tensors so it never imports IP-IQA internals:

  - ``base_output``:  [B, 2] from IP-IQA's original head
                      ([:, 0] = perceptual quality, [:, 1] = image-text alignment)
  - ``spatial_feat``: [B, C, H, W] = CLIP RN50 spatial feature inside IP-IQA

and returns the final [B, 2] output, where only the quality channel is fused:
    q_final = (1 - g) * q_base + g * q_local,   g = sigmoid(gate_logit)
The alignment channel is passed through untouched.
"""

import torch
import torch.nn as nn

from .local_distortion import LocalDistortionBranch


class GatedLocalFusion(nn.Module):
    """Stitch module: base head output + lightweight local quality branch.

    Args:
        in_channels: channel dim of IP-IQA's spatial feature (2048 for CLIP RN50).
        hidden_dim:  local branch hidden dim after the 1x1 projection (256).
        spatial_dim: N = H*W of the spatial feature. For 512 input with RN50
                     stride 32: (512//32)^2 = 256.
        gate_init:   initial gate logit. sigmoid(-2.0) ~= 0.119, so the model
                     starts trusting the original IP-IQA path (~88%) and lets
                     the gate learn how much to trust the local branch.
        drop:        dropout probability in the local branch heads.
        use_attention: keep the transpose-attention block (Full Ours) or drop it
                       (weighted-local ablation). See the plan, section 13.
    """

    def __init__(self, in_channels=2048, hidden_dim=256, spatial_dim=256,
                 gate_init=-2.0, drop=0.1, use_attention=True):
        super().__init__()
        self.local_branch = LocalDistortionBranch(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            drop=drop,
            use_attention=use_attention,
        )
        self.local_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, base_output, spatial_feat):
        # base_output: [B, 2]
        # spatial_feat: [B, in_channels, H, W]
        q_base = base_output[:, 0:1]
        align = base_output[:, 1:2]

        q_local = self.local_branch(spatial_feat)  # [B, 1]

        g = torch.sigmoid(self.local_gate_logit)
        q_final = (1.0 - g) * q_base + g * q_local

        return torch.cat([q_final, align], dim=1)
