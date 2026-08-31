"""Reader/writer for the final Blosc-packed proton BEV cache."""

from __future__ import annotations

import os
from pathlib import Path

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset


ARRAY_NAMES = ("ct", "dose", "energy")


def write_cache_sample(
    path: Path,
    ct: np.ndarray,
    dose: np.ndarray,
    energy: np.ndarray,
    compression_level: int = 5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = {
        name: np.frombuffer(
            blosc2.pack_array(
                array,
                clevel=compression_level,
                codec=blosc2.Codec.ZSTD,
                filter=blosc2.Filter.BITSHUFFLE,
            ),
            dtype=np.uint8,
        )
        for name, array in zip(ARRAY_NAMES, (ct, dose, energy))
    }
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **packed)
    os.replace(temporary, path)


def read_cache_sample(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as packed:
            missing = set(ARRAY_NAMES) - set(packed.files)
            if missing:
                raise ValueError(f"missing arrays: {sorted(missing)}")
            arrays = tuple(blosc2.unpack_array(packed[name].tobytes()) for name in ARRAY_NAMES)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unreadable proton cache sample: {path}") from error
    return arrays


class ProtonCacheDataset(Dataset):
    """Load CT, scaled dose and normalized energy from one cache split."""

    def __init__(self, root: str | Path, lateral_flip_probability: float = 0.0) -> None:
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*/*.npz"))
        if not self.paths:
            raise ValueError(f"No cached proton samples found under {self.root}")
        if not 0.0 <= lateral_flip_probability <= 1.0:
            raise ValueError("lateral_flip_probability must be in [0, 1]")
        self.lateral_flip_probability = float(lateral_flip_probability)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        ct, dose, energy = read_cache_sample(path)
        if ct.shape != dose.shape or ct.ndim != 3:
            raise ValueError(f"Invalid shapes in {path}: ct={ct.shape}, dose={dose.shape}")
        energy = np.asarray(energy, dtype=np.float32).reshape(-1)
        if energy.size != 1 or not np.isfinite(energy[0]):
            raise ValueError(f"Invalid energy in {path}: {energy}")
        input_tensor = torch.from_numpy(np.ascontiguousarray(ct[None].astype(np.float32)))
        target_tensor = torch.from_numpy(np.ascontiguousarray(dose[None].astype(np.float32)))
        if self.lateral_flip_probability and torch.rand(()) < self.lateral_flip_probability:
            input_tensor = torch.flip(input_tensor, dims=(2,))
            target_tensor = torch.flip(target_tensor, dims=(2,))
        return {
            "input": input_tensor,
            "target": target_tensor,
            "energy": torch.from_numpy(energy.copy()),
            "key": path.stem,
        }
