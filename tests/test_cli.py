import numpy as np

from dragnet_cmr.cli import main


def test_synthetic_cli(tmp_path):
    target = tmp_path / "synthetic.npz"
    preview = tmp_path / "preview.png"
    code = main([
        "synthetic",
        str(target),
        "--samples",
        "2",
        "--preview",
        str(preview),
    ])
    assert code == 0
    assert target.exists()
    assert preview.exists()
    with np.load(target, allow_pickle=False) as archive:
        assert archive["sequences"].shape == (2, 7, 128, 128)


def test_smoke_cli(capsys):
    code = main(["smoke", "--device", "cpu", "--seed", "5"])
    assert code == 0
    output = capsys.readouterr().out
    assert "1,218,253" in output
    assert "not a performance demo" in output
