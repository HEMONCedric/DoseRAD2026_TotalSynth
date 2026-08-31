#!/usr/bin/env python3
"""Predict scaled or physical BEV dose for one cached proton sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from doserad2026.cache import read_cache_sample
from doserad2026.model import ProtonFiLMLiteResUNet3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scaled", action="store_true", help="Keep the training-time x100000 scale")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ct, _target, energy = read_cache_sample(args.sample)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in state.items()
        if not key.removeprefix("_orig_mod.").startswith(("aux_dec2.", "aux_dec3."))
    }
    model = ProtonFiLMLiteResUNet3D(base_channels=6).to(device).eval()
    model.load_state_dict(state, strict=True)
    input_tensor = torch.from_numpy(ct[None, None].astype(np.float32)).to(device)
    energy_tensor = torch.from_numpy(np.asarray(energy, dtype=np.float32).reshape(1, 1)).to(device)
    with torch.inference_mode():
        prediction = model(input_tensor, energy_tensor)[0, 0].float().cpu().numpy()
    if not args.scaled:
        prediction /= 100_000.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, prediction.astype(np.float32))
    print(f"wrote {args.output} shape={prediction.shape}")


if __name__ == "__main__":
    main()
