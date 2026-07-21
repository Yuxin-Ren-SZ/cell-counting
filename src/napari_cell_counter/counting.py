"""Per-ROI result structures and label counting helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def count_labels(labels: np.ndarray) -> int:
    """Number of distinct non-zero label ids."""
    if labels.size == 0:
        return 0
    return int(np.unique(labels[labels > 0]).size)


@dataclass
class RoiResult:
    """Result for a single ROI."""

    roi_index: int
    count: int
    label_ids: list[int]  # offset ids as they appear in the composite layer
    bbox: tuple[int, int, int, int]


@dataclass
class RunResult:
    """Everything one segmentation run produced.

    ``labels`` is the full-size composite (2D or 3D). Replaces the ad-hoc
    ``last_run`` dict the old GUI stashed in a module global.
    """

    labels: np.ndarray
    per_roi: list[RoiResult]
    backend: str  # "local" | "remote"
    mode: str  # "single" | "double_positive_min" | "double_positive_coloc"
    run_meta: dict
    image_path: str | None = None
    roi_polygons: list[list] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(r.count for r in self.per_roi)
