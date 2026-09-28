import numpy as np

from dragnet_cmr.visualization import flow_to_rgb, save_registration_figure, save_sequence_preview


def test_flow_rgb_and_figures(tmp_path):
    flow = np.zeros((2, 16, 16), dtype=np.float32)
    rgb = flow_to_rgb(flow)
    assert rgb.shape == (16, 16, 3)

    sequence = np.zeros((2, 16, 16), dtype=np.float32)
    preview = save_sequence_preview(sequence, tmp_path / "preview.png")
    assert preview.exists()

    displacement = np.zeros((2, 2, 16, 16), dtype=np.float32)
    figure = save_registration_figure(
        sequence,
        sequence,
        displacement,
        tmp_path / "registration.png",
        border=1,
    )
    assert figure.exists()
