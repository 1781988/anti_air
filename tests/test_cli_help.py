from anti_air.cli import main


def test_clean_cache_help_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(tmp_path / "missing.yaml"), "clean-cache", "--cache", str(tmp_path / "cache")]) == 0
