#!/usr/bin/env python3
"""Export a training checkpoint as a compact inference-only state dictionary."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from doserad2026.model import ProtonFiLMLiteResUNet3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("The source checkpoint does not contain a state dictionary")
    exported = {
        key.removeprefix("_orig_mod."): value.detach().cpu()
        for key, value in state.items()
        if not key.removeprefix("_orig_mod.").startswith(("aux_dec2.", "aux_dec3."))
    }
    model = ProtonFiLMLiteResUNet3D(base_channels=6, deep_supervision=False)
    model.load_state_dict(exported, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
