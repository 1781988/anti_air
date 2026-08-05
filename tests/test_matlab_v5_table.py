from __future__ import annotations

import numpy as np

from anti_air.radar import _candidate_to_frame, numeric_radar_frame


def test_matlab_v5_table_raw_mapping_is_reconstructed() -> None:
    data = np.empty((1, 2), dtype=object)
    data[0, 0] = np.arange(5, dtype=float).reshape(-1, 1)
    data[0, 1] = (10.0 + np.arange(5, dtype=float)).reshape(-1, 1)

    raw_table = {
        "data": data,
        "varnames": np.array([["time", "snr"]], dtype=object),
        "nrows": np.array([[5.0]]),
        "nvars": np.array([[2.0]]),
        "rownames": np.empty((0, 0), dtype=object),
        "props": {"versionSavedFrom": np.array([[5.0]])},
    }

    frame = _candidate_to_frame(raw_table, "radar")
    assert frame is not None
    assert frame.shape == (5, 2)
    assert list(frame.columns) == ["time", "snr"]
    assert frame["time"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert frame["snr"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert frame.attrs["matlab_table_fallback"] is True

    numeric = numeric_radar_frame(frame)
    assert numeric.shape == (5, 2)
