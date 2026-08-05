from __future__ import annotations

from pathlib import Path

from anti_air.dataset import resolve_input_file


def test_resolve_nested_competition_inputs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data" / "train"
    nested = root / "初赛数据" / "科目一"
    nested.mkdir(parents=True)
    radar = nested / "radar_339_class-B_16：18.mat"
    infrared = nested / "ir_339_class-B_16：18.mp4"
    radar.write_bytes(b"radar")
    infrared.write_bytes(b"video")
    monkeypatch.chdir(tmp_path)

    guessed_radar = root / "radar_339_class-B_16:18.mat"
    guessed_ir = root / "ir_339_class-B_16:18.mp4"

    assert resolve_input_file(guessed_radar, modality="radar", batch_id="339") == radar.resolve()
    assert resolve_input_file(guessed_ir, modality="ir", batch_id="339") == infrared.resolve()


def test_resolve_input_file_reports_missing_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "data" / "train" / "ir_999.mp4"
    try:
        resolve_input_file(missing, modality="ir", batch_id="999")
    except FileNotFoundError as exc:
        message = str(exc)
        assert "could not be found recursively" in message
        assert "resolved_manifest.csv" in message
    else:
        raise AssertionError("Expected FileNotFoundError")
