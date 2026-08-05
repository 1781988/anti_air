import torch

from anti_air.model import build_model


def test_multimodal_forward() -> None:
    config = {"radar_channels": 8}
    model = build_model(config, num_classes=2, pretrained=False)
    output = model(
        torch.randn(2, 8, 64),
        torch.randn(2, 4, 3, 96, 96),
        torch.rand(2, 4),
    )
    assert output["logits"].shape == (2, 2)
    assert output["gates"].shape == (2, 2)
    assert torch.allclose(output["gates"].sum(dim=-1), torch.ones(2), atol=1e-5)
