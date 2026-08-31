#!/usr/bin/env python3
"""Download pinned public IMPACTSynth/KonfAI resources for offline use."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download


IMPACT_SYNTH_REPO = "VBoussot/ImpactSynth"
IMPACT_SYNTH_REVISION = "d00d991cd6a84683bf83f7cfe53630d7bdbb73b6"
IMPACT_SYNTH_CV1_SHA256 = "02c4e57011fad5890c23b1707040454c363883ccabcfffba19ec2480503c0905"
IMPACT_SEG_REPO = "VBoussot/ImpactSeg"
IMPACT_SEG_REVISION = "e34d956e27347d6621b27a087d5bfe0c87c89f5c"
IMPACT_SEG_BODY_SHA256 = "ba1c39f84a6128e6813e17f0f193330719fdb91a4d1aa77dce0c920cff78b4d2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/impact_synth"))
    parser.add_argument(
        "--without-body-model",
        action="store_true",
        help="Skip the public ImpactSeg body-mask dependency",
    )
    return parser.parse_args()


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {observed}")
    print(f"verified {path.name}: sha256={observed}")


def main() -> None:
    args = parse_args()
    hub_cache = args.output / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    impact_synth = snapshot_download(
        repo_id=IMPACT_SYNTH_REPO,
        revision=IMPACT_SYNTH_REVISION,
        cache_dir=hub_cache,
        allow_patterns=[
            "MR_CBCT/CV_1.pt",
            "MR_CBCT/Config.yml",
            "MR_CBCT/Evaluation.yml",
            "MR_CBCT/Model.py",
            "MR_CBCT/Prediction.yml",
            "MR_CBCT/UNetPlusPlus.yml",
            "MR_CBCT/Uncertainty.yml",
            "MR_CBCT/app.json",
            "MR_CBCT/requirements.txt",
        ],
    )
    synth_ref = hub_cache / "models--VBoussot--ImpactSynth" / "refs" / "main"
    synth_ref.parent.mkdir(parents=True, exist_ok=True)
    synth_ref.write_text(IMPACT_SYNTH_REVISION + "\n", encoding="utf-8")
    verify_sha256(Path(impact_synth) / "MR_CBCT" / "CV_1.pt", IMPACT_SYNTH_CV1_SHA256)
    print(f"IMPACTSynth CV_1 snapshot: {Path(impact_synth) / 'MR_CBCT'}")

    if not args.without_body_model:
        impact_seg = snapshot_download(
            repo_id=IMPACT_SEG_REPO,
            revision=IMPACT_SEG_REVISION,
            cache_dir=hub_cache,
            allow_patterns=["body/*"],
        )
        seg_ref = hub_cache / "models--VBoussot--ImpactSeg" / "refs" / "main"
        seg_ref.parent.mkdir(parents=True, exist_ok=True)
        seg_ref.write_text(IMPACT_SEG_REVISION + "\n", encoding="utf-8")
        verify_sha256(Path(impact_seg) / "body" / "Body.pt", IMPACT_SEG_BODY_SHA256)
        print(f"ImpactSeg body snapshot: {Path(impact_seg) / 'body'}")


if __name__ == "__main__":
    main()
