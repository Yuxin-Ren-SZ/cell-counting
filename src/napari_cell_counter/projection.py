"""Double-positive counting: min-projection and colocalization.

Two ways to count cells positive in channel A *and* channel B:

* :func:`min_projection` -- the method the user described for co-nuclear
  stains. Per-pixel minimum of the two channels keeps signal only where *both*
  are bright, collapsing ``C x Z x Y x X`` to ``Z x Y x X``. Segment that.
* :func:`colocalize` -- the general method. Segment A and B separately and
  keep A-cells whose mask sufficiently overlaps a B-cell. More robust to
  intensity mismatch and works for non-nuclear markers, at ~2x the seg cost.
"""
from __future__ import annotations

import numpy as np

from .axes import AxisModel, channel_plane


def _normalize(plane: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile-normalize to [0, 1] so a dim channel doesn't zero the min."""
    plane = plane.astype(np.float32)
    lo, hi = np.percentile(plane, [low, high])
    if hi <= lo:
        return np.zeros_like(plane)
    return np.clip((plane - lo) / (hi - lo), 0.0, 1.0)


def min_projection(
    model: AxisModel, chan_a: int, chan_b: int, normalize: bool = True
) -> np.ndarray:
    """Per-pixel minimum of two channels -> ``(Y,X)`` or ``(Z,Y,X)``.

    Channels are percentile-normalized first (unless ``normalize=False``) so
    differing exposures between the two stains don't make one channel dominate
    the minimum.
    """
    a = channel_plane(model, chan_a)
    b = channel_plane(model, chan_b)
    if normalize:
        a, b = _normalize(a), _normalize(b)
    return np.minimum(a, b)


def colocalize(
    labels_a: np.ndarray, labels_b: np.ndarray, min_overlap: float = 0.5
) -> np.ndarray:
    """Keep A-cells whose overlap fraction with any B-cell exceeds ``min_overlap``.

    Returns a labels array (same shape) containing only the surviving A-cells,
    with their original A ids preserved.
    """
    if labels_a.shape != labels_b.shape:
        raise ValueError("labels_a and labels_b must have the same shape")

    b_present = labels_b > 0
    keep_ids: list[int] = []
    for aid in np.unique(labels_a[labels_a > 0]):
        cell = labels_a == aid
        area = int(cell.sum())
        if area == 0:
            continue
        overlap = int(np.count_nonzero(cell & b_present))
        if overlap / area >= min_overlap:
            keep_ids.append(int(aid))

    return np.where(np.isin(labels_a, keep_ids), labels_a, 0)
