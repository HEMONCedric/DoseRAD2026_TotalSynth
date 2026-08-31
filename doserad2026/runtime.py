"""Grand Challenge proton CT/MRI inference runtime."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch

from .geometry import BevSpec, ImageGeometry, make_ray_frame, patient_to_bev, prefilter_image, bev_to_patient
from .mha import Mha4DWriter
from .model import ProtonFiLMLiteResUNet3D


NUM_OUTPUT_FILES = 10
PROTON_METADATA = "stacked-proton-beam-level-metadata.json"
ENERGY_MIN_MEV = 31.729
ENERGY_MAX_MEV = 200.7966
DOSE_SCALE = 100_000.0


@dataclass(frozen=True)
class BeamletJob:
    image_file_idx: int
    output_file_idx: int
    idx_in_output: int
    minimum_cutoff: float
    ray_source: tuple[float, float, float]
    ray_target: tuple[float, float, float]
    energy_mev: float

    @property
    def ray_key(self) -> tuple:
        return self.image_file_idx, self.ray_source, self.ray_target


@dataclass
class LoadedAnatomy:
    image: sitk.Image
    coefficients_xyz: np.ndarray
    geometry: ImageGeometry
    maximum_hu: float


def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    state = loaded.get("model", loaded)
    if not isinstance(state, dict):
        raise TypeError(f"No state dictionary in {path}")
    clean = {
        key.removeprefix("_orig_mod."): value
        for key, value in state.items()
        if not key.removeprefix("_orig_mod.").startswith(("aux_dec2.", "aux_dec3."))
    }
    return clean


class ProtonRuntime:
    def __init__(self) -> None:
        self.task = os.environ.get("TASK", "proton-ct")
        if self.task not in {"proton-ct", "proton-mri"}:
            raise ValueError("TASK must be proton-ct or proton-mri")
        self.input_base = (
            "radiation-dose-calculation-source-ct-image"
            if self.task == "proton-ct"
            else "radiation-dose-calculation-source-mri-image"
        )
        self.model_root = Path(os.environ.get("DOSERAD_MODEL_DIR", "/opt/ml/model"))
        self.weights = Path(
            os.environ.get(
                "DOSERAD_PROTON_WEIGHTS",
                self.model_root / "proton" / "proton_film_lite_b6_epoch79.pt",
            )
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = max(1, int(os.environ.get("DOSERAD_BATCH_SIZE", "16")))
        self.use_amp = os.environ.get("DOSERAD_USE_AMP", "1").lower() not in {"0", "false", "no"}
        self.model: ProtonFiLMLiteResUNet3D | None = None
        self.spec = BevSpec()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Missing proton weights: {self.weights}. See models/README.md."
            )
        model = ProtonFiLMLiteResUNet3D(base_channels=6, deep_supervision=False)
        model.load_state_dict(_load_weights(self.weights), strict=True)
        self.model = model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last_3d)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        print(
            f"[DoseRAD] loaded {self.weights.name} on {self.device}; task={self.task}; "
            f"batch={self.batch_size}",
            flush=True,
        )

    def predict(self, input_path: Path, output_path: Path) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Runtime is not loaded")
        started = time.perf_counter()
        metadata = json.loads((input_path / PROTON_METADATA).read_text(encoding="utf-8"))
        jobs_by_output = self._parse_jobs(metadata)
        anatomy_cache: dict[int, LoadedAnatomy] = {}
        bev_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        total_jobs = 0

        for output_index, jobs in enumerate(jobs_by_output):
            output_file = (
                output_path
                / "images"
                / f"stacked-radiation-dose-map-{output_index + 1}"
                / "output.mha"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if not jobs:
                sitk.WriteImage(sitk.Image(1, 1, sitk.sitkFloat32), str(output_file))
                continue
            image_indices = {job.image_file_idx for job in jobs}
            if len(image_indices) != 1:
                raise ValueError(f"Output slot {output_index} references multiple images")
            image_index = next(iter(image_indices))
            if image_index not in anatomy_cache:
                anatomy_cache[image_index] = self._load_anatomy(input_path, image_index)
            anatomy = anatomy_cache[image_index]
            jobs = sorted(jobs, key=lambda job: job.idx_in_output)
            stack_size = max(job.idx_in_output for job in jobs) + 1
            with Mha4DWriter(output_file, anatomy.image, stack_size) as writer:
                for offset in range(0, len(jobs), self.batch_size):
                    batch = jobs[offset : offset + self.batch_size]
                    predictions = self._predict_batch(anatomy, batch, bev_cache)
                    writer.write([job.idx_in_output for job in batch], predictions)
                    total_jobs += len(batch)

        report = {"seconds": time.perf_counter() - started, "beamlets": total_jobs}
        print(f"[DoseRAD][TIMING] {json.dumps(report)}", flush=True)
        return report

    @torch.inference_mode()
    def _predict_batch(
        self,
        anatomy: LoadedAnatomy,
        jobs: list[BeamletJob],
        bev_cache: OrderedDict[tuple, np.ndarray],
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Runtime is not loaded")
        volumes = []
        energies = []
        for job in jobs:
            key = job.ray_key
            if key not in bev_cache:
                frame = make_ray_frame(job.ray_source, job.ray_target)
                hu = patient_to_bev(
                    anatomy.coefficients_xyz,
                    anatomy.geometry,
                    frame,
                    self.spec,
                    cval=-1024.0,
                    clip=(-1024.0, anatomy.maximum_hu),
                )
                bev_cache[key] = ((np.clip(hu, -1024.0, 3000.0) + 1024.0) / 4024.0).astype(
                    np.float32
                )
                while len(bev_cache) > 16:
                    bev_cache.popitem(last=False)
            bev_cache.move_to_end(key)
            volumes.append(bev_cache[key])
            energy = (job.energy_mev - ENERGY_MIN_MEV) / (ENERGY_MAX_MEV - ENERGY_MIN_MEV)
            energies.append(np.clip(energy, 0.0, 1.0))

        tensor = torch.from_numpy(np.stack(volumes)[:, None]).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        if self.device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last_3d)
        energy_tensor = torch.tensor(energies, dtype=torch.float32, device=self.device)[:, None]
        autocast = (
            torch.autocast(
                "cuda",
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            if self.device.type == "cuda" and self.use_amp
            else nullcontext()
        )
        with autocast:
            prediction_bev = self.model(tensor, energy_tensor)
        prediction_bev = prediction_bev[:, 0].float().cpu().numpy() / DOSE_SCALE

        outputs = []
        for prediction, job in zip(prediction_bev, jobs):
            patient = bev_to_patient(
                prediction,
                anatomy.geometry,
                make_ray_frame(job.ray_source, job.ray_target),
                self.spec,
            )
            patient[patient <= job.minimum_cutoff] = 0.0
            outputs.append(patient)
        return np.stack(outputs).astype(np.float32, copy=False)

    def _load_anatomy(self, input_path: Path, image_index: int) -> LoadedAnatomy:
        location = input_path / "images" / f"{self.input_base}-{image_index + 1}"
        files = sorted(glob.glob(str(location / "*.mha")))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one .mha under {location}, found {len(files)}")
        source = Path(files[0])
        if self.task == "proton-mri":
            source = self._synthesize_ct(source, image_index)
        image = sitk.ReadImage(str(source), sitk.sitkFloat32)
        coefficients, geometry, maximum = prefilter_image(image)
        return LoadedAnatomy(image, coefficients, geometry, maximum)

    def _synthesize_ct(self, mr_path: Path, image_index: int) -> Path:
        precomputed = os.environ.get("DOSERAD_PRECOMPUTED_SCT_DIR")
        if precomputed:
            path = Path(precomputed) / f"sct-{image_index + 1}.mha"
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        output = Path("/tmp") / f"doserad-sct-{os.getpid()}-{image_index + 1}.mha"
        command = [
            "python",
            "/opt/app/scripts/predict_sct_public.py",
            str(mr_path),
            str(output),
            "--model-cache",
            str(self.model_root / "impact_synth"),
            "--gpu",
            str(self.device.index or 0),
        ]
        subprocess.run(command, check=True)
        return output

    @staticmethod
    def _parse_jobs(metadata: list[dict[str, Any]]) -> list[list[BeamletJob]]:
        outputs: list[list[BeamletJob]] = [[] for _ in range(NUM_OUTPUT_FILES)]
        seen: set[tuple[int, int]] = set()
        for image in metadata:
            image_index = int(image["image_file_idx"])
            for beam in image["beams"]:
                for ray in beam["rays"]:
                    source = tuple(float(value) for value in ray["ray_source"])
                    target = tuple(float(value) for value in ray["ray_target"])
                    for beamlet in ray["beamlets"]:
                        output_info = beamlet["output_info"]
                        output_index = int(output_info["output_file_idx"])
                        stack_index = int(output_info["idx_in_output"])
                        key = (output_index, stack_index)
                        if key in seen:
                            raise ValueError(f"Duplicate output location {key}")
                        seen.add(key)
                        if not 0 <= output_index < NUM_OUTPUT_FILES:
                            raise ValueError(f"Invalid output_file_idx={output_index}")
                        cutoff = float(output_info["minimum_cutoff"])
                        if not np.isfinite(cutoff):
                            raise ValueError(f"Non-finite minimum_cutoff at {key}")
                        outputs[output_index].append(
                            BeamletJob(
                                image_file_idx=image_index,
                                output_file_idx=output_index,
                                idx_in_output=stack_index,
                                minimum_cutoff=cutoff,
                                ray_source=source,
                                ray_target=target,
                                energy_mev=float(beamlet["energy"]),
                            )
                        )
        return outputs
