from pathlib import Path
import json

import numpy as np
import SimpleITK as sitk
import torch

from doserad2026.cache import read_cache_sample, write_cache_sample
from doserad2026.geometry import BevSpec
from doserad2026.runtime import ProtonRuntime


def test_blosc_cache_roundtrip(tmp_path: Path):
    ct = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    dose = (ct / 10).astype(np.float16)
    energy = np.asarray([0.25], dtype=np.float32)
    path = tmp_path / "patient" / "sample.npz"
    write_cache_sample(path, ct, dose, energy)
    observed = read_cache_sample(path)
    np.testing.assert_array_equal(observed[0], ct)
    np.testing.assert_array_equal(observed[1], dose)
    np.testing.assert_array_equal(observed[2], energy)


def test_proton_metadata_parser():
    metadata = [
        {
            "image_file_idx": 0,
            "beams": [
                {
                    "rays": [
                        {
                            "ray_source": [-100.0, 0.0, 0.0],
                            "ray_target": [0.0, 0.0, 0.0],
                            "beamlets": [
                                {
                                    "energy": 100.0,
                                    "output_info": {
                                        "output_file_idx": 2,
                                        "idx_in_output": 7,
                                        "minimum_cutoff": 1.0e-5,
                                    },
                                }
                            ],
                        }
                    ]
                }
            ],
        }
    ]
    outputs = ProtonRuntime._parse_jobs(metadata)
    assert len(outputs) == 10
    assert len(outputs[2]) == 1
    assert outputs[2][0].idx_in_output == 7
    assert outputs[2][0].energy_mev == 100.0


def test_invoke_io_contract_with_a_dummy_network(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TASK", "proton-ct")
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    image_dir = input_root / "images" / "radiation-dose-calculation-source-ct-image-1"
    image_dir.mkdir(parents=True)
    image = sitk.GetImageFromArray(np.zeros((5, 5, 5), dtype=np.float32))
    sitk.WriteImage(image, str(image_dir / "input.mha"))
    metadata = [
        {
            "image_file_idx": 0,
            "beams": [
                {
                    "rays": [
                        {
                            "ray_source": [-10.0, 2.0, 2.0],
                            "ray_target": [2.0, 2.0, 2.0],
                            "beamlets": [
                                {
                                    "energy": 100.0,
                                    "output_info": {
                                        "output_file_idx": 0,
                                        "idx_in_output": 0,
                                        "minimum_cutoff": 0.0,
                                    },
                                }
                            ],
                        }
                    ]
                }
            ],
        }
    ]
    (input_root / "stacked-proton-beam-level-metadata.json").write_text(json.dumps(metadata))

    class ZeroNetwork(torch.nn.Module):
        def forward(self, inputs, energy):
            return torch.zeros_like(inputs)

    runtime = ProtonRuntime()
    runtime.model = ZeroNetwork()
    runtime.device = torch.device("cpu")
    runtime.batch_size = 1
    runtime.spec = BevSpec((8, 4, 4), (1.0, 1.0, 1.0), 3.0)
    report = runtime.predict(input_root, output_root)
    assert report["beamlets"] == 1
    dose = sitk.ReadImage(
        str(output_root / "images" / "stacked-radiation-dose-map-1" / "output.mha")
    )
    assert dose.GetDimension() == 4
    assert dose.GetSize() == (5, 5, 5, 1)
    for slot in range(2, 11):
        assert (
            output_root / "images" / f"stacked-radiation-dose-map-{slot}" / "output.mha"
        ).is_file()
