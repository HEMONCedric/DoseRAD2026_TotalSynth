#!/usr/bin/env python3
"""Run the pinned public IMPACTSynth MR_CBCT CV-1 model on one MR volume."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from huggingface_hub import hf_hub_download
from ruamel.yaml import YAML
import torch
import torch.nn.functional as F


IMPACT_SYNTH_REPO = "VBoussot/ImpactSynth"
IMPACT_SYNTH_REVISION = "d00d991cd6a84683bf83f7cfe53630d7bdbb73b6"
IMPACT_SEG_REPO = "VBoussot/ImpactSeg"
IMPACT_SEG_REVISION = "e34d956e27347d6621b27a087d5bfe0c87c89f5c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input MR MetaImage")
    parser.add_argument("output", type=Path, help="Output sCT MetaImage")
    parser.add_argument("--model-cache", type=Path, default=Path("models/impact_synth"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--keep-work-dir", type=Path)
    return parser.parse_args()


def cached_file(
    cache_root: Path,
    filename: str,
    *,
    repo_id: str = IMPACT_SYNTH_REPO,
    revision: str = IMPACT_SYNTH_REVISION,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_root / "hub",
            local_files_only=True,
        )
    )


def rename_transform(mapping, old: str, new: str) -> None:
    """Rename a transform without changing its execution order."""

    index = list(mapping).index(old)
    value = mapping.pop(old)
    mapping.insert(index, new, value)


def render_body_config(
    source: Path, destination: Path, dataset: Path, model_definition: Path
) -> None:
    yaml = YAML()
    config = yaml.load(source.read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["classpath"] = str(model_definition.absolute())
    predictor_dataset = config["Predictor"]["Dataset"]
    predictor_dataset["dataset_filenames"] = [f"{dataset}:mha"]
    predictor_dataset["num_workers"] = 0
    transforms = predictor_dataset["groups_src"]["Volume_0"]["groups_dest"]["Volume"][
        "transforms"
    ]
    rename_transform(transforms, "Resample", "ResampleToResolution")
    with destination.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)


def render_config(source: Path, destination: Path, dataset: Path, model_definition: Path) -> None:
    """Render the public sCT config with an explicit precomputed body mask."""

    yaml = YAML()
    config = yaml.load(source.read_text(encoding="utf-8"))
    # Do not resolve this Hugging Face snapshot symlink: its blob target has no
    # `.yml` suffix, which prevents KonfAI from recognizing a declarative model.
    config["Predictor"]["Model"]["classpath"] = str(model_definition.absolute())
    predictor_dataset = config["Predictor"]["Dataset"]
    predictor_dataset["dataset_filenames"] = [f"{dataset}:mha"]
    # Batch one accepts arbitrary scanner matrix sizes whose final padded edge
    # patch can differ in shape; batching does not alter model semantics.
    predictor_dataset["batch_size"] = 1
    predictor_dataset["num_workers"] = 0
    groups = predictor_dataset["groups_src"]
    volume_destinations = groups["Volume_0"]["groups_dest"]
    # KonfAI 1.6 split the former concrete Resample into explicit transforms.
    # Run ImpactSeg directly, then expose its result as a normal MASK group;
    # this avoids the optional, unpublished `konfai-apps` nested runner.
    del volume_destinations["MASK"]
    rename_transform(
        volume_destinations["Volume"]["transforms"],
        "Resample",
        "ResampleToResolution",
    )
    groups["MASK"] = {
        "groups_dest": {
            "MASK": {
                "transforms": "None",
                "patch_transforms": "None",
                "is_input": False,
            }
        }
    }
    with destination.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)


def prepare_body_mask(
    source: Path,
    destination: Path,
    target_spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 3.0),
    dilation_radius: int = 5,
) -> None:
    """Apply IMPACTSynth's nearest resampling and box dilation to a body mask."""

    image = sitk.ReadImage(str(source))
    array_zyx = sitk.GetArrayFromImage(image)
    input_spacing_zyx = np.asarray(image.GetSpacing()[::-1], dtype=np.float64)
    target_spacing_zyx = np.asarray(target_spacing_xyz[::-1], dtype=np.float64)
    output_size_zyx = tuple(
        int(value)
        for value in (np.asarray(array_zyx.shape) * input_spacing_zyx / target_spacing_zyx)
    )
    tensor = torch.from_numpy((array_zyx > 0).astype(np.float32))[None, None]
    tensor = F.interpolate(tensor, size=output_size_zyx, mode="nearest")
    if dilation_radius:
        radius = int(dilation_radius)
        kernel = 2 * radius + 1
        tensor = F.max_pool3d(tensor, (kernel, 1, 1), stride=1, padding=(radius, 0, 0))
        tensor = F.max_pool3d(tensor, (1, kernel, 1), stride=1, padding=(0, radius, 0))
        tensor = F.max_pool3d(tensor, (1, 1, kernel), stride=1, padding=(0, 0, radius))
    output = sitk.GetImageFromArray(tensor[0, 0].to(torch.uint8).numpy())
    output.SetOrigin(image.GetOrigin())
    output.SetDirection(image.GetDirection())
    output.SetSpacing(target_spacing_xyz)
    sitk.WriteImage(output, str(destination))


def run_prediction(
    config: Path,
    checkpoint: Path,
    predictions: Path,
    gpu: int,
    work_dir: Path,
    environment: dict[str, str],
) -> None:
    command = [
        "konfai",
        "PREDICTION",
        "-y",
        "--gpu",
        str(gpu),
        "--config",
        str(config),
        "--models",
        str(checkpoint),
        "--predictions-dir",
        str(predictions),
    ]
    subprocess.run(command, cwd=work_dir, env=environment, check=True)


def find_output(predictions: Path, label: str) -> Path:
    candidates = [
        path
        for path in predictions.rglob("*.mha")
        if path.name.lower() in {"output.mha", "sct.mha"}
    ]
    if len(candidates) != 1:
        observed = [str(path) for path in predictions.rglob("*.mha")]
        raise RuntimeError(
            f"Expected one {label} Output.mha, found candidates={candidates}; all={observed}"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    config = cached_file(args.model_cache, "MR_CBCT/Prediction.yml")
    checkpoint = cached_file(args.model_cache, "MR_CBCT/CV_1.pt")
    model_definition = cached_file(args.model_cache, "MR_CBCT/UNetPlusPlus.yml")
    body_config = cached_file(
        args.model_cache,
        "body/Prediction.yml",
        repo_id=IMPACT_SEG_REPO,
        revision=IMPACT_SEG_REVISION,
    )
    body_checkpoint = cached_file(
        args.model_cache,
        "body/Body.pt",
        repo_id=IMPACT_SEG_REPO,
        revision=IMPACT_SEG_REVISION,
    )
    body_definition = cached_file(
        args.model_cache,
        "body/ResidualEncoderUNet.yml",
        repo_id=IMPACT_SEG_REPO,
        revision=IMPACT_SEG_REVISION,
    )

    temporary = None
    if args.keep_work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="doserad-sct-")
        work_dir = Path(temporary.name)
    else:
        work_dir = args.keep_work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    source_image = sitk.ReadImage(str(args.input))
    body_dataset = work_dir / "BodyDataset"
    body_case = body_dataset / "case"
    body_case.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(source_image, str(body_case / "Volume_0.mha"))
    rendered_body = work_dir / "BodyPrediction.yml"
    render_body_config(body_config, rendered_body, body_dataset.resolve(), body_definition)

    environment = os.environ.copy()
    environment["HF_HUB_CACHE"] = str((args.model_cache / "hub").resolve())
    environment["HF_HUB_OFFLINE"] = "1"
    body_predictions = work_dir / "BodyPredictions"
    run_prediction(
        rendered_body,
        body_checkpoint,
        body_predictions,
        args.gpu,
        work_dir,
        environment,
    )

    dataset = work_dir / "Dataset"
    case_dir = dataset / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(source_image, str(case_dir / "Volume_0.mha"))
    prepare_body_mask(find_output(body_predictions, "body mask"), case_dir / "MASK.mha")
    rendered = work_dir / "Prediction.yml"
    render_config(config, rendered, dataset.resolve(), model_definition)
    predictions = work_dir / "Predictions"
    run_prediction(rendered, checkpoint, predictions, args.gpu, work_dir, environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(find_output(predictions, "sCT"), args.output)
    print(f"wrote {args.output}")
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
