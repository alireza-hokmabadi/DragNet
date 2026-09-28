import numpy as np

from dragnet_cmr.metrics import jacobian_determinant_2d, nonpositive_jacobian_fraction, rmse


def test_rmse_zero_for_equal_images():
    image = np.ones((8, 8), dtype=np.float32)
    assert rmse(image, image) == 0.0


def test_zero_displacement_has_unit_jacobian():
    flow = np.zeros((2, 16, 16), dtype=np.float32)
    determinant = jacobian_determinant_2d(flow)
    assert np.allclose(determinant, 1.0)
    assert nonpositive_jacobian_fraction(flow) == 0.0
