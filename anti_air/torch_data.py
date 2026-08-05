from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import CachedRecord


@dataclass(frozen=True)
class WindowRef:
    path: Path
    index: int
    batch_id: str
    label: str


def make_refs(records: Iterable[CachedRecord]) -> list[WindowRef]:
    refs: list[WindowRef] = []
    for record in records:
        refs.extend(
            WindowRef(record.path, index, record.batch_id, record.label)
            for index in range(record.windows)
        )
    return refs


class WindowDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        refs: list[WindowRef],
        label_to_index: dict[str, int],
        *,
        augment: bool,
        seed: int,
        cache_files: int = 2,
    ) -> None:
        self.refs = refs
        self.label_to_index = label_to_index
        self.augment = augment
        self.seed = seed
        self.cache_files = max(1, cache_files)
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.refs)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path in self._cache:
            payload = self._cache.pop(path)
            self._cache[path] = payload
            return payload
        with np.load(path, allow_pickle=False) as handle:
            payload = {
                "radar": handle["radar"],
                "infrared": handle["infrared"],
                "quality": handle["quality"],
            }
        self._cache[path] = payload
        while len(self._cache) > self.cache_files:
            self._cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        payload = self._load(ref.path)
        radar = payload["radar"][ref.index].astype(np.float32)
        infrared = payload["infrared"][ref.index].astype(np.float32) / 255.0
        quality = payload["quality"][ref.index].astype(np.float32)

        if self.augment:
            rng = np.random.default_rng(self.seed + index + np.random.randint(0, 1_000_000))
            if rng.random() < 0.5:
                infrared = infrared[..., ::-1].copy()
            if rng.random() < 0.4:
                infrared[0] = np.clip(infrared[0] * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0, 1)
            if rng.random() < 0.35:
                radar += rng.normal(0.0, 0.03, size=radar.shape).astype(np.float32)
            if rng.random() < 0.25:
                channel = int(rng.integers(0, radar.shape[0]))
                radar[channel] = 0.0
            if rng.random() < 0.2:
                length = max(1, radar.shape[1] // 12)
                start = int(rng.integers(0, max(1, radar.shape[1] - length)))
                radar[:, start : start + length] = 0.0

        infrared = (infrared - 0.5) / 0.5
        return {
            "radar": torch.from_numpy(radar),
            "infrared": torch.from_numpy(infrared),
            "quality": torch.from_numpy(quality),
            "label": torch.tensor(self.label_to_index[ref.label], dtype=torch.long),
            "batch_id": ref.batch_id,
            "window_index": ref.index,
        }
