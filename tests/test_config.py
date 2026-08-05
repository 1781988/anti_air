from pathlib import Path

from anti_air.config import load_config


def test_quick_profile(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("profile: quick\n", encoding="utf-8")
    config = load_config(path)
    assert config["profile"] == "quick"
    assert config["epochs"] == 2
