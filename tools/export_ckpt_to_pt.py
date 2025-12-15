#!/usr/bin/env python3
"""Export a Lightning checkpoint to a plain model state_dict (.pt) for lazy inference.

Example:
  python tools/export_ckpt_to_pt.py \
    --ckpt runs/your_run/checkpoints/epoch=02.ckpt \
    --out weights/v9-c.pt \
    --model v9-c

The script composes the repo Hydra config to create the same model architecture
then loads the checkpoint's state dict (handles common Lightning prefixes) and
saves a clean `state_dict` suitable for `weight=` in `yolo/lazy.py`.
"""
#### TODO -> make it work!
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.model.yolo import create_model


def strip_prefixes(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        new_k = k
        for p in ("model.", "state_dict.", "module."):
            if new_k.startswith(p):
                new_k = new_k[len(p) :]
        out[new_k] = v
    return out


def main():
    p = argparse.ArgumentParser(description="Export Lightning checkpoint to .pt state_dict")
    p.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt file")
    p.add_argument("--out", required=True, help="Output .pt path to write state_dict")
    p.add_argument("--model", default="v9-c", help="Model name to use from config (e.g. v9-c)")
    p.add_argument("--class-num", type=int, default=None, help="Optional dataset class number override")
    p.add_argument("--override", nargs="*", default=[], help="Additional Hydra overrides (key=value)")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    # Load the model config YAML directly (avoid Hydra here). The model YAML
    # contains `name`, `anchor` and `model` fields required by `create_model`.
    model_cfg_path = project_root / "yolo" / "config" / "model" / f"{args.model}.yaml"
    if not model_cfg_path.exists():
        raise SystemExit(f"Model config not found: {model_cfg_path}")

    model_cfg = OmegaConf.load(str(model_cfg_path))
    class_num = args.class_num if args.class_num is not None else 80

    # load checkpoint early so we can infer class num if needed
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and ("model" in ckpt):
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        # could already be a state_dict
        state = ckpt
    else:
        raise SystemExit("Unsupported checkpoint format")

    state = strip_prefixes(state)

    # If user didn't pass class_num, try to infer from checkpoint (class conv weights)
    if args.class_num is None:
        inferred = None
        for k, v in state.items():
            if k.endswith(".class_conv.2.weight"):
                inferred = v.shape[0]
                break
        if inferred is not None:
            class_num = int(inferred)
        else:
            class_num = class_num

    # create model with correct class_num
    model = create_model(model_cfg, class_num=class_num, weight_path=None)

    # Try loading state_dict. If size mismatches raise, filter incompatible keys.
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        if not filtered:
            raise SystemExit(f"No compatible weights found to load. Error: {e}")
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Filtered out {len(skipped)} incompatible keys (e.g. class-head mismatch)")
    # save the cleaned state_dict
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))

    print(f"Saved cleaned state_dict to: {out_path}")
    if missing:
        print(f"Warning: missing keys: {len(missing)} (first 10): {missing[:10]}")
    if unexpected:
        print(f"Warning: unexpected keys: {len(unexpected)} (first 10): {unexpected[:10]}")


if __name__ == "__main__":
    main()
