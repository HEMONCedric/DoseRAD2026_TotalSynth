"""Patient/BEV geometry shared by cache preparation and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk


@dataclass(frozen=True)
class BevSpec:
    """Final proton grid in network order (depth, lateral, superior)."""

    shape_dls: tuple[int, int, int] = (288, 64, 64)
    spacing_dls: tuple[float, float, float] = (2.0, 1.0, 1.0)
    upstream_mm: float = 320.0


@dataclass(frozen=True)
class ImageGeometry:
    origin_xyz: np.ndarray
    spacing_xyz: np.ndarray
    direction: np.ndarray
    size_xyz: np.ndarray

    @classmethod
    def from_image(cls, image: sitk.Image) -> "ImageGeometry":
        if image.GetDimension() != 3:
            raise ValueError(f"Expected a 3-D image, got {image.GetDimension()}-D")
        return cls(
            origin_xyz=np.asarray(image.GetOrigin(), dtype=np.float64),
            spacing_xyz=np.asarray(image.GetSpacing(), dtype=np.float64),
            direction=np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3),
            size_xyz=np.asarray(image.GetSize(), dtype=np.int64),
        )

    def physical_to_continuous_index(self, points_xyz: np.ndarray) -> np.ndarray:
        relative = np.asarray(points_xyz, dtype=np.float64) - self.origin_xyz
        return (relative @ self.direction) / self.spacing_xyz

    def continuous_index_to_physical(self, index_xyz: np.ndarray) -> np.ndarray:
        scaled = np.asarray(index_xyz, dtype=np.float64) * self.spacing_xyz
        return self.origin_xyz + scaled @ self.direction.T


@dataclass(frozen=True)
class RayFrame:
    target_xyz: np.ndarray
    depth_axis: np.ndarray
    lateral_axis: np.ndarray
    superior_axis: np.ndarray

    @property
    def axes_dls(self) -> np.ndarray:
        return np.stack((self.depth_axis, self.lateral_axis, self.superior_axis), axis=0)


def normalize(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return array / norm


def make_ray_frame(ray_source, ray_target) -> RayFrame:
    """Build a right-handed depth/lateral/superior frame from a proton ray."""

    source = np.asarray(ray_source, dtype=np.float64)
    target = np.asarray(ray_target, dtype=np.float64)
    depth = normalize(target - source)
    patient_superior = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(depth @ patient_superior)) > 0.95:
        patient_superior = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    lateral = normalize(np.cross(patient_superior, depth))
    superior = normalize(np.cross(depth, lateral))
    return RayFrame(target, depth, lateral, superior)


def bev_world_points(frame: RayFrame, spec: BevSpec) -> np.ndarray:
    """Return all BEV voxel centers in an array shaped ``D,L,S,3``."""

    nd, nl, ns = spec.shape_dls
    sd, sl, ss = spec.spacing_dls
    depth = np.arange(nd, dtype=np.float64) * sd - spec.upstream_mm
    lateral = (np.arange(nl, dtype=np.float64) - (nl - 1) / 2.0) * sl
    superior = (np.arange(ns, dtype=np.float64) - (ns - 1) / 2.0) * ss
    dd, ll, ss_grid = np.meshgrid(depth, lateral, superior, indexing="ij")
    return (
        frame.target_xyz
        + dd[..., None] * frame.depth_axis
        + ll[..., None] * frame.lateral_axis
        + ss_grid[..., None] * frame.superior_axis
    )


def prefilter_image(image: sitk.Image) -> tuple[np.ndarray, ImageGeometry, float]:
    """Read a SimpleITK image and create cubic B-spline coefficients in X,Y,Z order."""

    volume_xyz = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0)).astype(np.float32)
    coefficients = ndi.spline_filter(volume_xyz, order=3, output=np.float32)
    return coefficients, ImageGeometry.from_image(image), float(volume_xyz.max())


def patient_to_bev(
    coefficients_xyz: np.ndarray,
    geometry: ImageGeometry,
    frame: RayFrame,
    spec: BevSpec,
    *,
    cval: float,
    clip: tuple[float, float] | None = None,
) -> np.ndarray:
    nd, nl, ns = spec.shape_dls
    sd, sl, ss = spec.spacing_dls
    axes = frame.axes_dls
    world_to_index = np.diag(1.0 / geometry.spacing_xyz) @ geometry.direction.T
    matrix = world_to_index @ axes.T @ np.diag((sd, sl, ss))
    physical_offset = np.asarray(
        (-spec.upstream_mm, -(nl - 1) * sl / 2.0, -(ns - 1) * ss / 2.0),
        dtype=np.float64,
    )
    offset = world_to_index @ (
        frame.target_xyz - geometry.origin_xyz + axes.T @ physical_offset
    )
    sampled = ndi.affine_transform(
        coefficients_xyz,
        matrix=matrix,
        offset=offset,
        output_shape=(nd, nl, ns),
        order=3,
        mode="constant",
        cval=float(cval),
        prefilter=False,
    )
    if clip is not None:
        sampled = np.clip(sampled, *clip)
    return sampled.astype(np.float32, copy=False)


def bev_to_patient(
    volume_dls: np.ndarray,
    patient_geometry: ImageGeometry,
    frame: RayFrame,
    spec: BevSpec,
) -> np.ndarray:
    """Cubic inverse resampling from BEV to a native image array in Z,Y,X order."""

    nx, ny, nz = (int(value) for value in patient_geometry.size_xyz)
    _nd, nl, ns = spec.shape_dls
    sd, sl, ss = spec.spacing_dls
    index_zyx_to_world = patient_geometry.direction @ np.diag(patient_geometry.spacing_xyz)
    index_zyx_to_world = index_zyx_to_world[:, [2, 1, 0]]
    axes = frame.axes_dls
    inverse_bev_spacing = np.diag((1.0 / sd, 1.0 / sl, 1.0 / ss))
    matrix = inverse_bev_spacing @ axes @ index_zyx_to_world
    offset = inverse_bev_spacing @ axes @ (
        patient_geometry.origin_xyz - frame.target_xyz
    ) + np.asarray((spec.upstream_mm / sd, (nl - 1) / 2.0, (ns - 1) / 2.0))
    sampled_zyx = ndi.affine_transform(
        np.asarray(volume_dls, dtype=np.float32),
        matrix=matrix,
        offset=offset,
        output_shape=(nz, ny, nx),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    return np.maximum(sampled_zyx, 0.0).astype(np.float32, copy=False)
