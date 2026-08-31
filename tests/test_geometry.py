import numpy as np
import SimpleITK as sitk

from doserad2026.geometry import (
    BevSpec,
    ImageGeometry,
    bev_to_patient,
    make_ray_frame,
    patient_to_bev,
    prefilter_image,
)


def make_linear_image() -> sitk.Image:
    z, y, x = np.meshgrid(
        np.arange(5, dtype=np.float32),
        np.arange(5, dtype=np.float32),
        np.arange(5, dtype=np.float32),
        indexing="ij",
    )
    image = sitk.GetImageFromArray(x + 10 * y + 100 * z)
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def test_patient_to_bev_axis_convention():
    image = make_linear_image()
    coefficients, geometry, _maximum = prefilter_image(image)
    frame = make_ray_frame((-1.0, 2.0, 2.0), (2.0, 2.0, 2.0))
    spec = BevSpec(shape_dls=(3, 3, 3), spacing_dls=(1.0, 1.0, 1.0), upstream_mm=1.0)
    bev = patient_to_bev(
        coefficients,
        geometry,
        frame,
        spec,
        cval=-1024.0,
    )
    # depth follows +X, lateral +Y and superior +Z for this horizontal ray.
    expected = np.empty((3, 3, 3), dtype=np.float32)
    for depth in range(3):
        for lateral in range(3):
            for superior in range(3):
                x = 1 + depth
                y = 1 + lateral
                z = 1 + superior
                expected[depth, lateral, superior] = x + 10 * y + 100 * z
    np.testing.assert_allclose(bev, expected, atol=1.0e-3)


def test_bev_to_patient_respects_native_zyx_order():
    image = make_linear_image()
    geometry = ImageGeometry.from_image(image)
    frame = make_ray_frame((-1.0, 2.0, 2.0), (2.0, 2.0, 2.0))
    spec = BevSpec(shape_dls=(5, 5, 5), spacing_dls=(1.0, 1.0, 1.0), upstream_mm=2.0)
    depth, lateral, superior = np.meshgrid(
        np.arange(5), np.arange(5), np.arange(5), indexing="ij"
    )
    bev = depth + 10 * lateral + 100 * superior
    patient = bev_to_patient(bev.astype(np.float32), geometry, frame, spec)
    expected = sitk.GetArrayFromImage(image)
    np.testing.assert_allclose(patient[1:-1, 1:-1, 1:-1], expected[1:-1, 1:-1, 1:-1], atol=1.0e-3)
