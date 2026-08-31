#!/usr/bin/env python3
"""Convert raw DoseRAD proton patients into the final Blosc BEV cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from doserad2026.cache import write_cache_sample
from doserad2026.geometry import (
    BevSpec,
    ImageGeometry,
    make_ray_frame,
    patient_to_bev,
    prefilter_image,
)


ENERGY_MIN_MEV = 31.729
ENERGY_MAX_MEV = 200.7966
DOSE_SCALE = 100_000.0
FINAL_VALIDATION = {
    "1ABB039",
    "1ABB042",
    "1ABB061",
    "1THB008",
    "1THB120",
    "1THB221",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-patients",
        nargs="*",
        default=sorted(FINAL_VALIDATION),
        help="Patient IDs written to validation; all other patients go to train",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-patients", type=int, help="Smoke-test helper")
    parser.add_argument("--limit-samples-per-patient", type=int, help="Smoke-test helper")
    return parser.parse_args()


def find_patients(root: Path) -> list[Path]:
    patients = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and (path / "image" / "ct.mha").is_file()
        and (path / f"{path.name}.json").is_file()
    ]
    if not patients:
        raise FileNotFoundError(f"No complete proton patient found under {root}")
    return patients


def normalized_energy(energy_mev: float) -> np.ndarray:
    value = (float(energy_mev) - ENERGY_MIN_MEV) / (ENERGY_MAX_MEV - ENERGY_MIN_MEV)
    if not -1.0e-5 <= value <= 1.00001:
        raise ValueError(f"Energy {energy_mev:g} MeV is outside the training range")
    return np.asarray([np.clip(value, 0.0, 1.0)], dtype=np.float32)


def dose_to_bev(path: Path, frame, spec: BevSpec) -> np.ndarray:
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    array_xyz = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0)).astype(np.float32)
    maximum = float(array_xyz.max())
    coefficients = ndi.spline_filter(array_xyz, order=3, output=np.float32)
    sampled = patient_to_bev(
        coefficients,
        ImageGeometry.from_image(image),
        frame,
        spec,
        cval=0.0,
        clip=(0.0, maximum),
    )
    return (sampled * DOSE_SCALE).astype(np.float16)


def process_patient(
    patient_dir: Path,
    output_root: Path,
    validation: set[str],
    overwrite: bool,
    sample_limit: int | None,
) -> int:
    patient_id = patient_dir.name
    split = "validation" if patient_id in validation else "train"
    destination = output_root / split / patient_id
    metadata = json.loads((patient_dir / f"{patient_id}.json").read_text(encoding="utf-8"))
    ct_image = sitk.ReadImage(str(patient_dir / "image" / "ct.mha"), sitk.sitkFloat32)
    ct_coefficients, ct_geometry, ct_maximum = prefilter_image(ct_image)
    spec = BevSpec()
    written = 0

    for beam in metadata["beams"]:
        beam_index = int(beam["beam_idx"])
        for ray in beam["rays"]:
            ray_index = int(ray["ray_idx"])
            frame = make_ray_frame(ray["ray_source"], ray["ray_target"])
            ct_bev = patient_to_bev(
                ct_coefficients,
                ct_geometry,
                frame,
                spec,
                cval=-1024.0,
                clip=(-1024.0, ct_maximum),
            )
            ct_bev = ((np.clip(ct_bev, -1024.0, 3000.0) + 1024.0) / 4024.0).astype(
                np.float16
            )
            for beamlet in ray["beamlets"]:
                beamlet_index = int(beamlet["beamlet_idx"])
                key = f"{patient_id}_B{beam_index}_R{ray_index}_L{beamlet_index}"
                output_path = destination / f"{key}.npz"
                if output_path.exists() and not overwrite:
                    written += 1
                    continue
                dose_path = (
                    patient_dir
                    / "dose"
                    / f"Dose_B{beam_index}_R{ray_index}_L{beamlet_index}.mha"
                )
                if not dose_path.is_file():
                    raise FileNotFoundError(dose_path)
                write_cache_sample(
                    output_path,
                    ct_bev,
                    dose_to_bev(dose_path, frame, spec),
                    normalized_energy(float(beamlet["energy"])),
                )
                written += 1
                if sample_limit is not None and written >= sample_limit:
                    return written
    return written


def main() -> None:
    args = parse_args()
    patients = find_patients(args.raw_root)
    if args.limit_patients is not None:
        patients = patients[: args.limit_patients]
    validation = set(args.validation_patients)
    observed = {patient.name for patient in patients}
    missing = validation - observed
    if missing and args.limit_patients is None:
        raise ValueError(f"Validation patients missing from raw root: {sorted(missing)}")

    args.output.mkdir(parents=True, exist_ok=True)
    total = 0
    for index, patient in enumerate(patients, start=1):
        count = process_patient(
            patient,
            args.output,
            validation,
            args.overwrite,
            args.limit_samples_per_patient,
        )
        total += count
        print(f"[{index}/{len(patients)}] {patient.name}: {count} samples", flush=True)
    print(f"cache complete: {total} samples in {args.output}")


if __name__ == "__main__":
    main()
