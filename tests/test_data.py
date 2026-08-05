from pathlib import Path

from anti_air.data import discover_samples, parse_filename


def test_parse_competition_filename() -> None:
    meta = parse_filename("ir_339_class-B_16：18.mp4")
    assert meta == {
        "modality": "ir",
        "batch_id": "339",
        "label": "class-B",
        "start_time": "16:18",
    }


def test_discover_direct_pairs(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "ir_1_class-A_10：00.mp4").write_bytes(b"x")
    (train / "radar_1_class-A_10：00.mat").write_bytes(b"x")
    nested = train / "nested"
    nested.mkdir()
    (nested / "ir_2_class-B_10：01.mp4").write_bytes(b"x")
    (nested / "radar_2_class-B_10：01.mat").write_bytes(b"x")
    samples = discover_samples(train)
    assert len(samples) == 1
    assert samples[0].batch_id == "1"
