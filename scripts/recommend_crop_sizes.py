#!/usr/bin/env python3
"""Recommend aligned 3D crop sizes for DoseRAD2026 from training doses.

The script uses:
- body-mask line traversal for the longitudinal extent
- thresholded dose support for transverse extents

Photon:
- target spacing defaults to 2 x 2 x 2 mm
- beam direction is derived from the CP gantry angle

Proton:
- target spacing defaults to 1 x 1 x 3 mm
- ray direction is derived from ray_source -> ray_target

The goal is to provide data-driven crop sizes that are small, but still cover
the impacted volume around the beam / ray axis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable
import time

import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend 3D crop sizes from training doses.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Dataset",
        help="Path to the DoseRAD2026 Dataset directory.",
    )
    parser.add_argument(
        "--dose-rel-threshold",
        type=float,
        default=1e-4,
        help="Relative threshold used to prefilter dose voxels before extent estimation.",
    )
    parser.add_argument(
        "--extent-mode",
        choices=("support", "energy"),
        default="energy",
        help="support: bbox from thresholded support; energy: smallest square x/y crop and z interval containing a target dose fraction.",
    )
    parser.add_argument(
        "--energy-fraction",
        type=float,
        default=0.95,
        help="Target integrated dose fraction kept by the crop in energy mode.",
    )
    parser.add_argument(
        "--body-threshold-hu",
        type=float,
        default=-900.0,
        help="HU threshold used to define the body mask from the CT.",
    )
    parser.add_argument(
        "--margin-xy-mm",
        type=float,
        default=10.0,
        help="Extra transverse margin added on each side of the crop.",
    )
    parser.add_argument(
        "--margin-z-mm",
        type=float,
        default=10.0,
        help="Extra longitudinal margin added at the entry and exit.",
    )
    parser.add_argument(
        "--line-step-mm",
        type=float,
        default=1.0,
        help="Sampling step used along the beam / ray axis for body traversal.",
    )
    parser.add_argument(
        "--photon-spacing",
        type=float,
        nargs=3,
        default=(2.0, 2.0, 2.0),
        metavar=("SX", "SY", "SZ"),
        help="Target spacing in mm for photon crops.",
    )
    parser.add_argument(
        "--proton-spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 3.0),
        metavar=("SX", "SY", "SZ"),
        help="Target spacing in mm for proton crops.",
    )
    parser.add_argument(
        "--photon-cp-samples",
        type=int,
        default=5,
        help="Number of uniformly spaced control points sampled per photon beam.",
    )
    parser.add_argument(
        "--proton-ray-samples",
        type=int,
        default=3,
        help="Number of uniformly spaced rays sampled per proton beam. All beamlets are kept.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Optional limit on the number of cases per modality.",
    )
    parser.add_argument(
        "--max-support-points",
        type=int,
        default=250_000,
        help="Maximum number of thresholded dose voxels transformed per sample.",
    )
    parser.add_argument(
        "--voxel-multiple",
        type=int,
        default=16,
        help="Round the suggested crop sizes up to this voxel multiple.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("photon", "proton"),
        default=("photon", "proton"),
        help="Modalities to process.",
    )
    parser.add_argument(
        "--record-file",
        type=Path,
        default=Path("recommend_crop_sizes_records.jsonl"),
        help="Append-only JSONL log with one record per processed contribution.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("recommend_crop_sizes_summary.json"),
        help="JSON summary checkpoint updated during the run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing record file by skipping already processed samples.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=200,
        help="Flush pending sample records to disk every N processed samples.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N processed samples.",
    )
    return parser.parse_args()


def uniform_indices(length: int, samples: int) -> list[int]:
    if length <= 0:
        return []
    if samples >= length:
        return list(range(length))
    if samples == 1:
        return [length // 2]
    return sorted({round(i * (length - 1) / (samples - 1)) for i in range(samples)})


def normalize(vector: Iterable[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return array / norm


def build_axes(z_axis: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(z_axis, up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.vstack([x_axis, y_axis, z_axis])


def photon_direction_from_gantry(gantry_angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(gantry_angle_deg)
    return normalize([math.sin(angle_rad), math.cos(angle_rad), 0.0])


class ImageGeometry:
    def __init__(self, image: sitk.Image) -> None:
        self.origin = np.asarray(image.GetOrigin(), dtype=np.float64)
        self.spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        self.direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
        self.size_xyz = np.asarray(image.GetSize(), dtype=np.int32)
        self.index_to_world = self.direction @ np.diag(self.spacing)
        self.world_to_index = np.diag(1.0 / self.spacing) @ self.direction.T
        self.corners_world = self._corners_world()

    def _corners_world(self) -> np.ndarray:
        nx, ny, nz = self.size_xyz.tolist()
        max_corner = np.array([nx - 1, ny - 1, nz - 1], dtype=np.float64)
        corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [max_corner[0], 0.0, 0.0],
                [0.0, max_corner[1], 0.0],
                [0.0, 0.0, max_corner[2]],
                [max_corner[0], max_corner[1], 0.0],
                [max_corner[0], 0.0, max_corner[2]],
                [0.0, max_corner[1], max_corner[2]],
                [max_corner[0], max_corner[1], max_corner[2]],
            ],
            dtype=np.float64,
        )
        return self.index_to_physical(corners)

    def index_to_physical(self, index_xyz: np.ndarray) -> np.ndarray:
        return self.origin + index_xyz @ self.index_to_world.T

    def physical_to_continuous_index(self, points_world: np.ndarray) -> np.ndarray:
        return (points_world - self.origin) @ self.world_to_index.T


def nearest_sample(mask_zyx: np.ndarray, geometry: ImageGeometry, points_world: np.ndarray) -> np.ndarray:
    index_xyz = np.rint(geometry.physical_to_continuous_index(points_world)).astype(np.int32)
    inside = np.all((index_xyz >= 0) & (index_xyz < geometry.size_xyz[None, :]), axis=1)
    values = np.zeros(len(points_world), dtype=bool)
    if np.any(inside):
        idx = index_xyz[inside]
        values[inside] = mask_zyx[idx[:, 2], idx[:, 1], idx[:, 0]]
    return values


def line_body_extent_mm(
    body_mask_zyx: np.ndarray,
    geometry: ImageGeometry,
    axis_point_world: np.ndarray,
    z_axis: np.ndarray,
    step_mm: float,
) -> tuple[float, float, float] | None:
    projections = (geometry.corners_world - axis_point_world) @ z_axis
    t_min = projections.min() - 20.0
    t_max = projections.max() + 20.0
    t = np.arange(t_min, t_max + step_mm, step_mm, dtype=np.float64)
    points_world = axis_point_world[None, :] + t[:, None] * z_axis[None, :]
    inside_body = nearest_sample(body_mask_zyx, geometry, points_world)
    if not np.any(inside_body):
        return None
    first = int(np.argmax(inside_body))
    last = int(len(inside_body) - 1 - np.argmax(inside_body[::-1]))
    entry_t = float(t[first])
    exit_t = float(t[last])
    return entry_t, exit_t, exit_t - entry_t


def dose_support_extents_mm(
    dose_zyx: np.ndarray,
    geometry: ImageGeometry,
    axis_point_world: np.ndarray,
    basis_xyz: np.ndarray,
    rel_threshold: float,
    max_support_points: int,
) -> tuple[float, float, float, float] | None:
    max_dose = float(np.max(dose_zyx))
    if max_dose <= 0.0:
        return None

    mask = dose_zyx >= (rel_threshold * max_dose)
    support_zyx = np.argwhere(mask)
    if support_zyx.size == 0:
        return None

    if len(support_zyx) > max_support_points:
        stride = math.ceil(len(support_zyx) / max_support_points)
        support_zyx = support_zyx[::stride]

    support_xyz = support_zyx[:, ::-1].astype(np.float64)
    support_world = geometry.index_to_physical(support_xyz)
    local_xyz = (support_world - axis_point_world[None, :]) @ basis_xyz.T
    half_x = float(np.max(np.abs(local_xyz[:, 0])))
    half_y = float(np.max(np.abs(local_xyz[:, 1])))
    z_min = float(np.min(local_xyz[:, 2]))
    z_max = float(np.max(local_xyz[:, 2]))
    return half_x, half_y, z_min, z_max


def shortest_weighted_interval(z_values: np.ndarray, weights: np.ndarray, target_weight: float) -> tuple[float, float, float] | None:
    if len(z_values) == 0:
        return None
    order = np.argsort(z_values)
    z_sorted = z_values[order]
    w_sorted = weights[order]
    left = 0
    accum = 0.0
    best: tuple[float, float, float] | None = None

    for right in range(len(z_sorted)):
        accum += float(w_sorted[right])
        while left < right and accum - float(w_sorted[left]) >= target_weight:
            accum -= float(w_sorted[left])
            left += 1
        if accum >= target_weight:
            z0 = float(z_sorted[left])
            z1 = float(z_sorted[right])
            length = z1 - z0
            if best is None or length < best[2]:
                best = (z0, z1, length)
    return best


def shortest_weighted_interval_hist(
    z_hist: np.ndarray,
    target_weight: float,
    z_origin_mm: float,
    z_spacing_mm: float,
) -> tuple[float, float, float] | None:
    if len(z_hist) == 0:
        return None

    left = 0
    accum = 0.0
    best: tuple[float, float, float] | None = None

    for right, weight in enumerate(z_hist):
        accum += float(weight)
        while left < right and accum - float(z_hist[left]) >= target_weight:
            accum -= float(z_hist[left])
            left += 1
        if accum >= target_weight:
            z_min = z_origin_mm + left * z_spacing_mm
            z_max = z_origin_mm + (right + 1) * z_spacing_mm
            length = z_max - z_min
            if best is None or length < best[2]:
                best = (z_min, z_max, length)
    return best


def dose_energy_extents_mm(
    dose_zyx: np.ndarray,
    geometry: ImageGeometry,
    axis_point_world: np.ndarray,
    basis_xyz: np.ndarray,
    rel_threshold: float,
    max_support_points: int,
    energy_fraction: float,
    transverse_spacing_mm: float,
    longitudinal_spacing_mm: float,
) -> tuple[float, float, float, float] | None:
    max_dose = float(np.max(dose_zyx))
    if max_dose <= 0.0:
        return None

    mask = dose_zyx >= (rel_threshold * max_dose)
    support_zyx = np.argwhere(mask)
    if support_zyx.size == 0:
        return None

    if len(support_zyx) > max_support_points:
        stride = math.ceil(len(support_zyx) / max_support_points)
        support_zyx = support_zyx[::stride]

    dose_values = dose_zyx[support_zyx[:, 0], support_zyx[:, 1], support_zyx[:, 2]].astype(np.float64)
    support_xyz = support_zyx[:, ::-1].astype(np.float64)
    support_world = geometry.index_to_physical(support_xyz)
    local_xyz = (support_world - axis_point_world[None, :]) @ basis_xyz.T

    chebyshev_radius = np.max(np.abs(local_xyz[:, :2]), axis=1)
    z_values = local_xyz[:, 2]
    total_weight = float(np.sum(dose_values))
    if total_weight <= 0.0:
        return None
    target_weight = energy_fraction * total_weight

    # Quantize to the target crop grid. This makes the search much faster than
    # re-sorting every subset radius-by-radius and matches the discrete nature
    # of the final output volume.
    radius_bins = np.ceil(chebyshev_radius / transverse_spacing_mm).astype(np.int32)
    z_origin = math.floor(float(np.min(z_values)) / longitudinal_spacing_mm) * longitudinal_spacing_mm
    z_bins = np.floor((z_values - z_origin) / longitudinal_spacing_mm).astype(np.int32)

    max_r = int(radius_bins.max())
    max_z = int(z_bins.max())
    num_z_bins = max_z + 1
    flat_index = radius_bins * num_z_bins + z_bins
    rz_hist = np.bincount(
        flat_index,
        weights=dose_values,
        minlength=(max_r + 1) * num_z_bins,
    ).reshape(max_r + 1, num_z_bins)
    cumulative_rz_hist = np.cumsum(rz_hist, axis=0)
    cumulative_mass = np.sum(cumulative_rz_hist, axis=1)

    best: tuple[float, float, float, float] | None = None
    first_radius = int(np.searchsorted(cumulative_mass, target_weight, side="left"))
    for radius_bin in range(first_radius, max_r + 1):
        z_hist = cumulative_rz_hist[radius_bin]
        interval = shortest_weighted_interval_hist(
            z_hist,
            target_weight,
            z_origin,
            longitudinal_spacing_mm,
        )
        if interval is None:
            continue

        radius = max(1, radius_bin) * transverse_spacing_mm
        z_min, z_max, z_len = interval
        z_len = max(z_len, longitudinal_spacing_mm)
        volume = (2.0 * radius) * (2.0 * radius) * z_len
        if best is None or volume < best[3]:
            best = (float(radius), z_min, z_max, float(volume))

    if best is None:
        return None

    radius, z_min, z_max, _ = best
    return radius, radius, z_min, z_max


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def round_up_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def mm_to_voxels(length_mm: float, spacing_mm: float, multiple: int) -> int:
    voxels = int(math.ceil(length_mm / spacing_mm))
    return round_up_to_multiple(max(voxels, 1), multiple)


def load_case_json(case_dir: Path) -> dict:
    case_json = case_dir / f"{case_dir.name}.json"
    return json.loads(case_json.read_text())


def empty_stats() -> dict[str, list[float]]:
    return {"body_length_mm": [], "dose_depth_mm": [], "half_x_mm": [], "half_y_mm": []}


def stats_counts(stats: dict[str, list[float]]) -> int:
    return len(stats["body_length_mm"])


def append_stats(stats: dict[str, list[float]], record: dict[str, object]) -> None:
    if not bool(record["valid"]):
        return
    stats["body_length_mm"].append(float(record["body_length_mm"]))
    stats["dose_depth_mm"].append(float(record["dose_depth_mm"]))
    stats["half_x_mm"].append(float(record["half_x_mm"]))
    stats["half_y_mm"].append(float(record["half_y_mm"]))


def count_planned_samples(args: argparse.Namespace, modality: str) -> int:
    root = args.dataset_root / modality / "training"
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if args.limit_cases is not None:
        case_dirs = case_dirs[: args.limit_cases]

    total = 0
    for case_dir in case_dirs:
        case_json = load_case_json(case_dir)
        if modality == "photon":
            for beam in case_json["beams"]:
                total += len(uniform_indices(len(beam["control_points"]), args.photon_cp_samples))
        else:
            for beam in case_json["beams"]:
                ray_indices = uniform_indices(len(beam["rays"]), args.proton_ray_samples)
                for ray_index in ray_indices:
                    total += len(beam["rays"][ray_index]["beamlets"])
    return total


def print_progress(
    modality: str,
    processed: int,
    planned: int | None,
    start_time: float,
    stats: dict[str, list[float]],
    initial_processed: int,
) -> None:
    elapsed = max(time.time() - start_time, 1e-9)
    session_processed = max(processed - initial_processed, 0)
    rate = session_processed / elapsed
    if planned and planned > 0:
        remaining = max(planned - processed, 0)
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        eta_hours = eta_seconds / 3600.0 if math.isfinite(eta_seconds) else float("inf")
        print(
            f"[{modality}] {processed}/{planned} processed | "
            f"{rate:.2f} samples/s | {stats_counts(stats)} valid | ETA {eta_hours:.2f} h",
            flush=True,
        )
    else:
        print(
            f"[{modality}] {processed} processed | "
            f"{rate:.2f} samples/s | {stats_counts(stats)} valid",
            flush=True,
        )


def flush_records(record_file: Path, pending_records: list[dict[str, object]]) -> None:
    if not pending_records:
        return
    record_file.parent.mkdir(parents=True, exist_ok=True)
    with record_file.open("a", encoding="utf-8") as handle:
        for record in pending_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    pending_records.clear()


def write_summary(
    summary_file: Path,
    args: argparse.Namespace,
    planned_counts: dict[str, int],
    processed_counts: dict[str, int],
    stats_by_modality: dict[str, dict[str, list[float]]],
) -> None:
    payload: dict[str, object] = {
        "args": {
            "dataset_root": str(args.dataset_root),
            "extent_mode": args.extent_mode,
            "energy_fraction": args.energy_fraction,
            "dose_rel_threshold": args.dose_rel_threshold,
            "photon_spacing": list(args.photon_spacing),
            "proton_spacing": list(args.proton_spacing),
            "photon_cp_samples": args.photon_cp_samples,
            "proton_ray_samples": args.proton_ray_samples,
        },
        "modalities": {},
    }

    for modality, stats in stats_by_modality.items():
        spacing = tuple(float(x) for x in (args.photon_spacing if modality == "photon" else args.proton_spacing))
        payload["modalities"][modality] = {
            "planned": planned_counts.get(modality, 0),
            "processed": processed_counts.get(modality, 0),
            "valid": stats_counts(stats),
            "summary_text": summarize_modality(
                modality.capitalize(),
                stats,
                spacing,
                args.voxel_multiple,
                args.margin_xy_mm,
                args.margin_z_mm,
            ),
        }

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_existing_records(record_file: Path) -> tuple[dict[str, dict[str, list[float]]], dict[str, set[str]], dict[str, int]]:
    stats_by_modality = {"photon": empty_stats(), "proton": empty_stats()}
    processed_keys = {"photon": set(), "proton": set()}
    processed_counts = {"photon": 0, "proton": 0}

    if not record_file.exists():
        return stats_by_modality, processed_keys, processed_counts

    with record_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            modality = str(record["modality"])
            sample_key = str(record["sample_key"])
            if sample_key in processed_keys[modality]:
                continue
            processed_keys[modality].add(sample_key)
            processed_counts[modality] += 1
            append_stats(stats_by_modality[modality], record)

    return stats_by_modality, processed_keys, processed_counts


def maybe_record_sample(
    record: dict[str, object],
    modality: str,
    args: argparse.Namespace,
    pending_records: list[dict[str, object]],
    processed_counts: dict[str, int],
    initial_counts: dict[str, int],
    planned_counts: dict[str, int],
    stats_by_modality: dict[str, dict[str, list[float]]],
    start_times: dict[str, float],
) -> None:
    pending_records.append(record)
    processed_counts[modality] += 1
    append_stats(stats_by_modality[modality], record)

    if args.progress_every > 0 and processed_counts[modality] % args.progress_every == 0:
        print_progress(
            modality,
            processed_counts[modality],
            planned_counts.get(modality),
            start_times[modality],
            stats_by_modality[modality],
            initial_counts.get(modality, 0),
        )

    if args.save_every > 0 and len(pending_records) >= args.save_every:
        flush_records(args.record_file, pending_records)
        write_summary(args.summary_file, args, planned_counts, processed_counts, stats_by_modality)


def summarize_modality(name: str, stats: dict[str, list[float]], spacing_xyz: tuple[float, float, float], multiple: int, margin_xy: float, margin_z: float) -> str:
    if not stats["body_length_mm"]:
        return f"{name}: no samples processed"

    body_p95 = percentile(stats["body_length_mm"], 95)
    body_p99 = percentile(stats["body_length_mm"], 99)
    dose_depth_p99 = percentile(stats["dose_depth_mm"], 99)
    half_x_p99 = percentile(stats["half_x_mm"], 99)
    half_y_p99 = percentile(stats["half_y_mm"], 99)

    crop_x_mm = 2.0 * (half_x_p99 + margin_xy)
    crop_y_mm = 2.0 * (half_y_p99 + margin_xy)
    crop_z_mm = max(body_p99 + 2.0 * margin_z, dose_depth_p99 + 2.0 * margin_z)

    nx = mm_to_voxels(crop_x_mm, spacing_xyz[0], multiple)
    ny = mm_to_voxels(crop_y_mm, spacing_xyz[1], multiple)
    nz = mm_to_voxels(crop_z_mm, spacing_xyz[2], multiple)
    nxy = max(nx, ny)

    lines = [
        f"{name}",
        f"  samples: {len(stats['body_length_mm'])}",
        f"  body length mm    p95={body_p95:.1f}  p99={body_p99:.1f}  max={max(stats['body_length_mm']):.1f}",
        f"  dose depth mm     p95={percentile(stats['dose_depth_mm'], 95):.1f}  p99={dose_depth_p99:.1f}  max={max(stats['dose_depth_mm']):.1f}",
        f"  half x mm         p95={percentile(stats['half_x_mm'], 95):.1f}  p99={half_x_p99:.1f}  max={max(stats['half_x_mm']):.1f}",
        f"  half y mm         p95={percentile(stats['half_y_mm'], 95):.1f}  p99={half_y_p99:.1f}  max={max(stats['half_y_mm']):.1f}",
        f"  recommended mm    x={crop_x_mm:.1f}  y={crop_y_mm:.1f}  z={crop_z_mm:.1f}",
        f"  recommended vox   x={nx}  y={ny}  z={nz}",
        f"  square xy vox     {nxy} x {nxy} x {nz}",
    ]
    return "\n".join(lines)


def process_photon(
    args: argparse.Namespace,
    stats: dict[str, list[float]],
    processed_keys: set[str],
    processed_counts: dict[str, int],
    initial_counts: dict[str, int],
    planned_counts: dict[str, int],
    pending_records: list[dict[str, object]],
    start_times: dict[str, float],
    stats_by_modality: dict[str, dict[str, list[float]]],
) -> None:
    root = args.dataset_root / "photon" / "training"
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if args.limit_cases is not None:
        case_dirs = case_dirs[: args.limit_cases]

    for case_dir in case_dirs:
        ct_img = sitk.ReadImage(str(case_dir / "image" / "ct.mha"))
        ct_arr = sitk.GetArrayViewFromImage(ct_img)
        ct_geo = ImageGeometry(ct_img)
        body_mask = ct_arr > args.body_threshold_hu
        case_json = load_case_json(case_dir)

        for beam in case_json["beams"]:
            iso_center = np.asarray(beam["iso_center"], dtype=np.float64)
            cp_indices = uniform_indices(len(beam["control_points"]), args.photon_cp_samples)
            for cp_index in cp_indices:
                cp = beam["control_points"][cp_index]
                sample_key = f"{case_dir.name}:B{beam['beam_idx']}:CP{cp['cp_idx']}"
                if sample_key in processed_keys:
                    continue
                z_axis = photon_direction_from_gantry(float(cp["gantry_angle"]))
                basis = build_axes(z_axis)
                body_extent = line_body_extent_mm(body_mask, ct_geo, iso_center, z_axis, args.line_step_mm)
                if body_extent is None:
                    maybe_record_sample(
                        {
                            "modality": "photon",
                            "sample_key": sample_key,
                            "valid": False,
                        },
                        "photon",
                        args,
                        pending_records,
                        processed_counts,
                        initial_counts,
                        planned_counts,
                        stats_by_modality,
                        start_times,
                    )
                    processed_keys.add(sample_key)
                    continue

                dose_path = case_dir / "dose" / f"Dose_B{beam['beam_idx']}_CP{cp['cp_idx']:03d}.mha"
                dose_img = sitk.ReadImage(str(dose_path))
                dose_arr = sitk.GetArrayViewFromImage(dose_img)
                dose_geo = ImageGeometry(dose_img)
                if args.extent_mode == "support":
                    dose_extent = dose_support_extents_mm(
                        dose_arr,
                        dose_geo,
                        iso_center,
                        basis,
                        args.dose_rel_threshold,
                        args.max_support_points,
                    )
                else:
                    dose_extent = dose_energy_extents_mm(
                        dose_arr,
                        dose_geo,
                        iso_center,
                        basis,
                        args.dose_rel_threshold,
                        args.max_support_points,
                        args.energy_fraction,
                        float(args.photon_spacing[0]),
                        float(args.photon_spacing[2]),
                    )
                if dose_extent is None:
                    maybe_record_sample(
                        {
                            "modality": "photon",
                            "sample_key": sample_key,
                            "valid": False,
                        },
                        "photon",
                        args,
                        pending_records,
                        processed_counts,
                        initial_counts,
                        planned_counts,
                        stats_by_modality,
                        start_times,
                    )
                    processed_keys.add(sample_key)
                    continue

                half_x, half_y, z_min, z_max = dose_extent
                maybe_record_sample(
                    {
                        "modality": "photon",
                        "sample_key": sample_key,
                        "valid": True,
                        "body_length_mm": body_extent[2],
                        "dose_depth_mm": z_max - z_min,
                        "half_x_mm": half_x,
                        "half_y_mm": half_y,
                    },
                    "photon",
                    args,
                    pending_records,
                    processed_counts,
                    initial_counts,
                    planned_counts,
                    stats_by_modality,
                    start_times,
                )
                processed_keys.add(sample_key)


def process_proton(
    args: argparse.Namespace,
    stats: dict[str, list[float]],
    processed_keys: set[str],
    processed_counts: dict[str, int],
    initial_counts: dict[str, int],
    planned_counts: dict[str, int],
    pending_records: list[dict[str, object]],
    start_times: dict[str, float],
    stats_by_modality: dict[str, dict[str, list[float]]],
) -> None:
    root = args.dataset_root / "proton" / "training"
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if args.limit_cases is not None:
        case_dirs = case_dirs[: args.limit_cases]

    for case_dir in case_dirs:
        ct_img = sitk.ReadImage(str(case_dir / "image" / "ct.mha"))
        ct_arr = sitk.GetArrayViewFromImage(ct_img)
        ct_geo = ImageGeometry(ct_img)
        body_mask = ct_arr > args.body_threshold_hu
        case_json = load_case_json(case_dir)

        for beam in case_json["beams"]:
            ray_indices = uniform_indices(len(beam["rays"]), args.proton_ray_samples)
            for ray_index in ray_indices:
                ray = beam["rays"][ray_index]
                ray_source = np.asarray(ray["ray_source"], dtype=np.float64)
                ray_target = np.asarray(ray["ray_target"], dtype=np.float64)
                z_axis = normalize(ray_target - ray_source)
                basis = build_axes(z_axis)
                body_extent = line_body_extent_mm(body_mask, ct_geo, ray_target, z_axis, args.line_step_mm)
                if body_extent is None:
                    for beamlet in ray["beamlets"]:
                        sample_key = (
                            f"{case_dir.name}:B{beam['beam_idx']}:R{ray['ray_idx']}:L{beamlet['beamlet_idx']}"
                        )
                        if sample_key in processed_keys:
                            continue
                        maybe_record_sample(
                            {
                                "modality": "proton",
                                "sample_key": sample_key,
                                "valid": False,
                            },
                            "proton",
                            args,
                            pending_records,
                            processed_counts,
                            initial_counts,
                            planned_counts,
                            stats_by_modality,
                            start_times,
                        )
                        processed_keys.add(sample_key)
                    continue

                for beamlet in ray["beamlets"]:
                    sample_key = (
                        f"{case_dir.name}:B{beam['beam_idx']}:R{ray['ray_idx']}:L{beamlet['beamlet_idx']}"
                    )
                    if sample_key in processed_keys:
                        continue
                    dose_path = case_dir / "dose" / (
                        f"Dose_B{beam['beam_idx']}_R{ray['ray_idx']}_L{beamlet['beamlet_idx']}.mha"
                    )
                    dose_img = sitk.ReadImage(str(dose_path))
                    dose_arr = sitk.GetArrayViewFromImage(dose_img)
                    dose_geo = ImageGeometry(dose_img)
                    if args.extent_mode == "support":
                        dose_extent = dose_support_extents_mm(
                            dose_arr,
                            dose_geo,
                            ray_target,
                            basis,
                            args.dose_rel_threshold,
                            args.max_support_points,
                        )
                    else:
                        dose_extent = dose_energy_extents_mm(
                            dose_arr,
                            dose_geo,
                            ray_target,
                            basis,
                            args.dose_rel_threshold,
                            args.max_support_points,
                            args.energy_fraction,
                            float(args.proton_spacing[0]),
                            float(args.proton_spacing[2]),
                        )
                    if dose_extent is None:
                        maybe_record_sample(
                            {
                                "modality": "proton",
                                "sample_key": sample_key,
                                "valid": False,
                            },
                            "proton",
                            args,
                            pending_records,
                            processed_counts,
                            initial_counts,
                            planned_counts,
                            stats_by_modality,
                            start_times,
                        )
                        processed_keys.add(sample_key)
                        continue

                    half_x, half_y, z_min, z_max = dose_extent
                    maybe_record_sample(
                        {
                            "modality": "proton",
                            "sample_key": sample_key,
                            "valid": True,
                            "body_length_mm": body_extent[2],
                            "dose_depth_mm": z_max - z_min,
                            "half_x_mm": half_x,
                            "half_y_mm": half_y,
                        },
                        "proton",
                        args,
                        pending_records,
                        processed_counts,
                        initial_counts,
                        planned_counts,
                        stats_by_modality,
                        start_times,
                    )
                    processed_keys.add(sample_key)


def main() -> None:
    args = parse_args()
    modalities = tuple(dict.fromkeys(args.modalities))

    stats_by_modality = {"photon": empty_stats(), "proton": empty_stats()}
    processed_keys = {"photon": set(), "proton": set()}
    processed_counts = {"photon": 0, "proton": 0}
    pending_records: list[dict[str, object]] = []

    if args.resume:
        stats_by_modality, processed_keys, processed_counts = load_existing_records(args.record_file)
        print(
            f"Resuming from {args.record_file}: "
            f"photon={processed_counts['photon']} processed, "
            f"proton={processed_counts['proton']} processed",
            flush=True,
        )
    else:
        if args.record_file.exists():
            args.record_file.unlink()
        if args.summary_file.exists():
            args.summary_file.unlink()

    planned_counts = {
        modality: count_planned_samples(args, modality)
        for modality in modalities
    }
    start_times = {modality: time.time() for modality in modalities}
    initial_counts = dict(processed_counts)

    try:
        if "photon" in modalities:
            process_photon(
                args,
                stats_by_modality["photon"],
                processed_keys["photon"],
                processed_counts,
                initial_counts,
                planned_counts,
                pending_records,
                start_times,
                stats_by_modality,
            )
            flush_records(args.record_file, pending_records)
            write_summary(args.summary_file, args, planned_counts, processed_counts, stats_by_modality)
            print(
                summarize_modality(
                    "Photon",
                    stats_by_modality["photon"],
                    tuple(float(x) for x in args.photon_spacing),
                    args.voxel_multiple,
                    args.margin_xy_mm,
                    args.margin_z_mm,
                ),
                flush=True,
            )
            print(flush=True)

        if "proton" in modalities:
            process_proton(
                args,
                stats_by_modality["proton"],
                processed_keys["proton"],
                processed_counts,
                initial_counts,
                planned_counts,
                pending_records,
                start_times,
                stats_by_modality,
            )
            flush_records(args.record_file, pending_records)
            write_summary(args.summary_file, args, planned_counts, processed_counts, stats_by_modality)
            print(
                summarize_modality(
                    "Proton",
                    stats_by_modality["proton"],
                    tuple(float(x) for x in args.proton_spacing),
                    args.voxel_multiple,
                    args.margin_xy_mm,
                    args.margin_z_mm,
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nInterrupted: flushing partial results to disk.", flush=True)
        flush_records(args.record_file, pending_records)
        write_summary(args.summary_file, args, planned_counts, processed_counts, stats_by_modality)
        for modality in modalities:
            print(
                summarize_modality(
                    modality.capitalize(),
                    stats_by_modality[modality],
                    tuple(float(x) for x in (args.photon_spacing if modality == "photon" else args.proton_spacing)),
                    args.voxel_multiple,
                    args.margin_xy_mm,
                    args.margin_z_mm,
                ),
                flush=True,
            )
            print(flush=True)
        raise
    finally:
        flush_records(args.record_file, pending_records)
        write_summary(args.summary_file, args, planned_counts, processed_counts, stats_by_modality)


if __name__ == "__main__":
    main()
