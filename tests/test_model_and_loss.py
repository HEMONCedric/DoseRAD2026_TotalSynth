import torch

from doserad2026.losses import Level1Loss
from doserad2026.model import ProtonFiLMLiteResUNet3D, count_parameters


def test_deployed_parameter_count_and_shape():
    model = ProtonFiLMLiteResUNet3D(base_channels=6, deep_supervision=False).eval()
    assert count_parameters(model) == 3_094_511
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 32, 32, 32), torch.tensor([[0.5]]))
    assert output.shape == (1, 1, 32, 32, 32)
    assert torch.all(output >= 0)


def test_deep_supervision_is_training_only():
    model = ProtonFiLMLiteResUNet3D(base_channels=6, deep_supervision=True).eval()
    with torch.inference_mode():
        main, dec2, dec3 = model(torch.zeros(1, 1, 32, 32, 32), torch.tensor([[0.5]]))
    assert main.shape == (1, 1, 32, 32, 32)
    assert dec2.shape == (1, 1, 16, 16, 16)
    assert dec3.shape == (1, 1, 8, 8, 8)


def test_level1_loss_is_zero_for_an_exact_prediction():
    target = torch.rand(2, 1, 8, 4, 4)
    terms = Level1Loss()(target.clone(), target)
    assert terms["masked_mae"].item() == 0.0
    assert abs(terms["idd"].item()) < 1.0e-7
    assert abs(terms["loss"].item()) < 1.0e-7
