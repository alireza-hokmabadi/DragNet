import torch

from dragnet_cmr.spatial import warp_image_2d


def test_zero_displacement_is_identity():
    image = torch.rand(2, 1, 32, 40)
    displacement = torch.zeros(2, 2, 32, 40)
    warped = warp_image_2d(image, displacement)
    assert torch.allclose(warped, image, atol=2e-6)


def test_shape_validation():
    image = torch.rand(1, 1, 32, 32)
    bad = torch.zeros(1, 3, 32, 32)
    try:
        warp_image_2d(image, bad)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
