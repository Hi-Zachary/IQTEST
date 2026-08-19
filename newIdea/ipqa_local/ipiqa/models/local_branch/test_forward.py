"""Standalone smoke test for the local_branch stitching module.

Runs on random tensors only -- no IP-IQA / AM-BQA code, no data needed.
Verifies shapes, the gated-fusion semantics (alignment untouched), the
no-attention ablation variant, and prints the parameter count.

Usage:
    python test_forward.py            # CPU
    python test_forward.py --cuda     # GPU if available
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from local_branch import GatedLocalFusion, LocalDistortionBranch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="try GPU")
    parser.add_argument("--in-channels", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--spatial", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    B = 2
    C, H, W = args.in_channels, args.spatial, args.spatial
    spatial_dim = H * W

    feat = torch.randn(B, C, H, W, device=device)
    base_output = torch.randn(B, 2, device=device)

    # 1) local branch alone
    branch = LocalDistortionBranch(
        in_channels=C, hidden_dim=args.hidden_dim, spatial_dim=spatial_dim
    ).to(device)
    q_local = branch(feat)
    print(f"[branch] q_local      shape={tuple(q_local.shape)}")
    assert q_local.shape == (B, 1)

    # 2) full gated fusion
    fused = GatedLocalFusion(
        in_channels=C, hidden_dim=args.hidden_dim, spatial_dim=spatial_dim
    ).to(device)
    out = fused(base_output, feat)
    print(f"[fused ] output       shape={tuple(out.shape)}")
    assert out.shape == (B, 2)
    assert torch.equal(out[:, 1], base_output[:, 1]), "alignment must be unchanged"
    print("[fused ] alignment unchanged: True")

    # 3) ablation: no attention (patch score/weight only)
    branch_noattn = LocalDistortionBranch(
        in_channels=C, hidden_dim=args.hidden_dim, spatial_dim=spatial_dim,
        use_attention=False,
    ).to(device)
    q_local2 = branch_noattn(feat)
    print(f"[ablat ] no-attn q_local shape={tuple(q_local2.shape)}")
    assert q_local2.shape == (B, 1)

    n_param = sum(p.numel() for p in fused.parameters())
    print(f"[fused ] params      {n_param:,}  (device={device})")
    print("ALL OK")


if __name__ == "__main__":
    main()
