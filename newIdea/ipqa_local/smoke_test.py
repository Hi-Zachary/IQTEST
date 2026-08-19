"""Smoke test: verify the stitching module is correctly integrated into IP-IQA.

Builds the model from a given config (E0/E1/E2), runs one forward pass on a
random image + prompt, and checks shapes. Requires a GPU because IP-IQA's
encode_text uses .cuda().

Usage (from project root):
    python smoke_test.py [path/to/config.yaml]
    # default: ipiqa/projects/agiqa3k/ipiqa_ours.yaml
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        "ipiqa/projects/agiqa3k/ipiqa_ours.yaml"

    import clip  # noqa: F401  (top-level vendored CLIP)
    import ipiqa  # noqa: F401  (registers models/tasks/processors)

    from omegaconf import OmegaConf
    from ipiqa.common.registry import registry

    assert torch.cuda.is_available(), "smoke test requires CUDA (encode_text hardcodes .cuda())"

    config = OmegaConf.load(cfg_path)
    model_cfg = config.model
    print(f"config: {cfg_path}")
    print(f"use_local_branch={model_cfg.get('use_local_branch', False)}, "
          f"local_use_attention={model_cfg.get('local_use_attention', True)}")

    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg).to("cuda")
    model.eval()

    x = torch.randn(1, 3, model_cfg.input_resolution, model_cfg.input_resolution,
                    device="cuda")
    text = ["a statue of a man"]

    with torch.no_grad():
        out = model(x, text)

    print(f"output shape: {tuple(out.shape)}")
    assert out.shape == (1, 2)

    if model.local_fusion is not None:
        n = sum(p.numel() for p in model.local_fusion.parameters())
        print(f"local_branch params: {n:,}")
    else:
        print("local_fusion: None (baseline path, alignment unchanged)")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
