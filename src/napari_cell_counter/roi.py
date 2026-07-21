"""ROI handling: crop-to-bbox before segmentation, then per-ROI reassembly.

The old pipeline sent the *whole* image to Cellpose and filtered cells by a
single union mask afterwards (``napari_gui.py:140-188``). Here each ROI shape
becomes its own bounding-box crop (so a smaller image is segmented) and each
ROI is counted separately; results are pasted back into one composite labels
array with per-ROI label-ID offsets so IDs never collide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.draw import polygon2mask


@dataclass
class RoiCrop:
    """One ROI: a bounding box (in full-image Y/X coords) + its polygon mask."""

    index: int
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1)
    polygon_mask: np.ndarray  # bool, shape (y1 - y0, x1 - x0)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1]


def full_frame_crop(image_yx: tuple[int, int]) -> RoiCrop:
    """The implicit single ROI covering the whole image (no ROI drawn)."""
    h, w = image_yx
    return RoiCrop(
        index=0,
        bbox=(0, 0, h, w),
        polygon_mask=np.ones((h, w), dtype=bool),
    )


def crops_from_shapes(shape_vertices, image_yx: tuple[int, int]) -> list[RoiCrop]:
    """Build a :class:`RoiCrop` per drawn shape.

    ``shape_vertices`` is an iterable of ``(N, 2)`` vertex arrays in ``(row,
    col)`` order (napari ``Shapes.data`` convention). Only the last two
    coordinates of each vertex are used, so shapes drawn on a multi-dim layer
    still work. Degenerate shapes (< 3 vertices or zero-area bbox) are skipped.
    """
    h, w = image_yx
    crops: list[RoiCrop] = []
    for verts in shape_vertices:
        verts = np.asarray(verts, dtype=float)
        if verts.ndim != 2 or verts.shape[0] < 3:
            continue
        yx = verts[:, -2:]  # last two axes are Y, X

        y0 = max(0, int(np.floor(yx[:, 0].min())))
        x0 = max(0, int(np.floor(yx[:, 1].min())))
        y1 = min(h, int(np.ceil(yx[:, 0].max())))
        x1 = min(w, int(np.ceil(yx[:, 1].max())))
        if y1 <= y0 or x1 <= x0:
            continue

        local = yx - np.array([y0, x0])
        mask = polygon2mask((y1 - y0, x1 - x0), local)
        if not mask.any():
            continue
        crops.append(
            RoiCrop(index=len(crops), bbox=(y0, x0, y1, x1), polygon_mask=mask)
        )
    return crops


def apply_polygon(labels: np.ndarray, polygon_mask: np.ndarray) -> np.ndarray:
    """Zero out labels outside the polygon within the crop.

    Works for 2D ``(Y,X)`` and 3D ``(Z,Y,X)`` label arrays (the mask is
    broadcast across Z). Returns a new array.
    """
    if labels.ndim == 2:
        return np.where(polygon_mask, labels, 0)
    if labels.ndim == 3:
        return np.where(polygon_mask[np.newaxis, :, :], labels, 0)
    raise ValueError(f"labels must be 2D or 3D, got {labels.ndim}D")


def paste_into(
    composite: np.ndarray,
    crop_labels: np.ndarray,
    bbox: tuple[int, int, int, int],
    id_offset: int,
) -> int:
    """Paste ``crop_labels`` (already polygon-masked) into ``composite``.

    Nonzero labels are shifted by ``id_offset`` so ROIs don't collide, then
    written with ``np.maximum`` (so overlapping ROI bboxes don't clobber each
    other). Returns the new running max label id.

    ``composite`` is the full-size labels array: ``(Y,X)`` for 2D or
    ``(Z,Y,X)`` for 3D. ``crop_labels`` matches in dimensionality.
    """
    y0, x0, y1, x1 = bbox
    shifted = np.where(crop_labels > 0, crop_labels + id_offset, 0)
    if composite.ndim == 2:
        region = composite[y0:y1, x0:x1]
        composite[y0:y1, x0:x1] = np.maximum(region, shifted)
    else:  # 3D
        region = composite[:, y0:y1, x0:x1]
        composite[:, y0:y1, x0:x1] = np.maximum(region, shifted)
    return int(max(id_offset, shifted.max()) if shifted.size else id_offset)
