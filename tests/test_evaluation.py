from pathlib import Path

from anti_air.evaluation import _folds
from anti_air.preprocess import CachedRecord


def record(batch: str, label: str) -> CachedRecord:
    return CachedRecord(batch, label, Path(f"{batch}.npz"), 1, {}, {}, {}, False)


def test_three_record_split_is_partial_diagnostic() -> None:
    records = [record("1", "A"), record("2", "A"), record("3", "B")]
    folds, strategy = _folds(records, 5, 2026)
    assert strategy == "partial_leave_one_record_out"
    assert len(folds) == 2
    assert all(test in ([0], [1]) for _, test in folds)
