"""Bounded-memory writer for Grand Challenge 4-D MetaImage dose stacks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


class Mha4DWriter:
    """Write frames directly into an uncompressed local-data `.mha` file."""

    def __init__(self, destination: Path, reference: sitk.Image, stack_size: int) -> None:
        self.destination = destination
        self.stack_size = int(stack_size)
        size_xyz = tuple(int(value) for value in reference.GetSize())
        shape_tzyx = (self.stack_size, size_xyz[2], size_xyz[1], size_xyz[0])
        direction = np.eye(4, dtype=np.float64)
        direction[:3, :3] = np.asarray(reference.GetDirection()).reshape(3, 3)
        header = (
            "ObjectType = Image\n"
            "NDims = 4\n"
            "BinaryData = True\n"
            "BinaryDataByteOrderMSB = False\n"
            "CompressedData = False\n"
            f"TransformMatrix = {_numbers(direction.T.reshape(-1))}\n"
            f"Offset = {_numbers((*reference.GetOrigin(), 0.0))}\n"
            "CenterOfRotation = 0 0 0 0\n"
            f"ElementSpacing = {_numbers((*reference.GetSpacing(), 1.0))}\n"
            f"DimSize = {' '.join(str(v) for v in (*size_xyz, self.stack_size))}\n"
            "AnatomicalOrientation = ????\n"
            "ElementType = MET_FLOAT\n"
            "ElementDataFile = LOCAL\n"
        ).encode("ascii")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._file = destination.open("w+b")
        self._file.write(header)
        self._offset = len(header)
        self._file.truncate(self._offset + int(np.prod(shape_tzyx)) * 4)
        self._array = np.memmap(
            self._file,
            dtype="<f4",
            mode="r+",
            offset=self._offset,
            shape=shape_tzyx,
            order="C",
        )

    def write(self, indices: list[int], volumes_zyx: np.ndarray) -> None:
        if volumes_zyx.shape[0] != len(indices):
            raise ValueError("Frame count and output index count differ")
        self._array[np.asarray(indices, dtype=np.int64)] = volumes_zyx

    def close(self) -> None:
        if getattr(self, "_array", None) is None:
            return
        self._array.flush()
        del self._array
        self._array = None
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
