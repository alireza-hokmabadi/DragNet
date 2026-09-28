import torch

from dragnet_cmr.model import DragNet


def test_parameter_count_matches_historical_architecture():
    model = DragNet()
    assert model.parameter_count() == 1_218_253


def test_forward_shapes_and_finite_loss():
    torch.manual_seed(0)
    model = DragNet().eval()
    sequence = torch.rand(1, 2, 1, 128, 128)
    with torch.no_grad():
        output = model(sequence)
    assert output.registered.shape == (1, 2, 1, 128, 128)
    assert output.displacement.shape == (1, 2, 2, 128, 128)
    assert torch.isfinite(output.losses.total)


def test_generation_shapes():
    torch.manual_seed(0)
    model = DragNet().eval()
    first = torch.rand(1, 1, 128, 128)
    second = torch.rand(1, 1, 128, 128)
    one = model.generate_from_one_frame(first, frame_count=2)
    two = model.generate_from_two_frames(first, second, frame_count=2)
    assert one.generated.shape == (1, 2, 1, 128, 128)
    assert one.displacement.shape == (1, 2, 2, 128, 128)
    assert two.generated.shape == one.generated.shape


def test_cholesky_displacement_sampling_shape():
    model = DragNet(displacement_sampling="cholesky")
    mu = torch.zeros(1, 2, 4, 4)
    covariance = torch.eye(2).reshape(1, 1, 1, 2, 2).expand(1, 4, 4, 2, 2).clone()
    sample = model.sample_displacement(mu, covariance)
    assert sample.shape == mu.shape
