from pathlib import Path

import pytest


pytest.importorskip("ruamel.yaml")

from ruamel.yaml import YAML  # noqa: E402

from scripts.predict_sct_public import prepare_body_mask, render_config  # noqa: E402


def test_public_sct_config_uses_absolute_local_resources(tmp_path: Path) -> None:
    source = tmp_path / "source.yml"
    source.write_text(
        "Predictor:\n"
        "  Model:\n"
        "    classpath: UNetPlusPlus.yml\n"
        "  Dataset:\n"
        "    groups_src:\n"
        "      Volume_0:\n"
        "        groups_dest:\n"
        "          MASK:\n"
        "            transforms: {KonfAIInference: {}}\n"
        "          Volume:\n"
        "            transforms:\n"
        "              Resample: {spacing: [1, 1, 3], inverse: true}\n"
        "    dataset_filenames: [./Dataset:mha]\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "Dataset"
    model_blob = tmp_path / "blob-without-extension"
    model_blob.write_text("name: test\n", encoding="utf-8")
    model = tmp_path / "UNetPlusPlus.yml"
    model.symlink_to(model_blob)
    destination = tmp_path / "rendered.yml"

    render_config(source, destination, dataset, model)
    config = YAML(typ="safe").load(destination.read_text(encoding="utf-8"))

    assert config["Predictor"]["Model"]["classpath"] == str(model.absolute())
    assert config["Predictor"]["Model"]["classpath"].endswith(".yml")
    assert config["Predictor"]["Dataset"]["dataset_filenames"] == [f"{dataset}:mha"]
    groups = config["Predictor"]["Dataset"]["groups_src"]
    assert "MASK" in groups
    assert "MASK" not in groups["Volume_0"]["groups_dest"]
    mask_transforms = groups["MASK"]["groups_dest"]["MASK"]["transforms"]
    assert mask_transforms == "None"
    transforms = groups["Volume_0"]["groups_dest"]["Volume"]["transforms"]
    assert "ResampleToResolution" in transforms


def test_body_mask_matches_public_resample_and_dilation(tmp_path: Path) -> None:
    import numpy as np
    import SimpleITK as sitk

    array = np.zeros((3, 4, 5), dtype=np.uint8)
    array[1, 2, 2] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((2.0, 2.0, 3.0))
    source = tmp_path / "body.mha"
    destination = tmp_path / "mask.mha"
    sitk.WriteImage(image, str(source))

    prepare_body_mask(source, destination, dilation_radius=0)
    observed = sitk.ReadImage(str(destination))

    assert observed.GetSize() == (10, 8, 3)
    assert observed.GetSpacing() == (1.0, 1.0, 3.0)
    assert set(np.unique(sitk.GetArrayFromImage(observed))) <= {0, 1}
