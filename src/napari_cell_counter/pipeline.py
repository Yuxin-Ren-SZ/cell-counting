"""Orchestration: crop -> (optional projection) -> segment -> per-ROI count.

Kept free of any napari/Qt import so it can be unit-tested with a fake
segmenter. The widget supplies the :class:`~napari_cell_counter.axes.AxisModel`
and the list of :class:`~napari_cell_counter.roi.RoiCrop` and calls
:func:`run_counting`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import axes, projection, roi
from .counting import RoiResult, RunResult

# Segmentation modes.
SINGLE = "single"
DP_MIN = "double_positive_min"
DP_COLOC = "double_positive_coloc"


@dataclass
class RunOptions:
    backend: str = "local"  # "local" | "remote"
    channel: int | None = None  # None == all channels (single mode)
    mode: str = SINGLE
    chan_a: int = 0
    chan_b: int = 1
    do_3D: bool = False
    coloc_min_overlap: float = 0.5
    min_size_refilter: int = 15  # drop polygon-clipped fragments below this
    hf_token: str | None = None
    local_params: dict = field(default_factory=dict)
    remote_params: dict = field(default_factory=dict)


def _drop_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Remove labels with fewer than ``min_size`` pixels (after polygon cut)."""
    if min_size is None or min_size <= 0:
        return labels
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    keep = ids[counts >= min_size]
    if keep.size == ids.size:
        return labels
    return np.where(np.isin(labels, keep), labels, 0)


def _default_local_segmenter(plane, *, channel_axis, do_3D, z_axis, **params):
    from .segmentation import segment_local

    return segment_local(
        plane, channel_axis=channel_axis, do_3D=do_3D, z_axis=z_axis, **params
    )


def _default_remote_segmenter(plane, *, out_shape, hf_token, **params):
    from .remote import segment_remote

    return segment_remote(plane, out_shape=out_shape, hf_token=hf_token, **params)


def _segment_plane(
    plane, channel_axis, opts: RunOptions, out_shape, do_3D, local_seg, remote_seg
):
    """Dispatch one array to the chosen backend; return a labels array."""
    if opts.backend == "remote":
        if plane.ndim != 2:
            raise ValueError(
                "Remote backend is 2D only. Select a single channel (not "
                "'All channels' on a multi-channel image) and disable 3D, "
                "or switch to the local backend."
            )
        labels, _count, _meta = remote_seg(
            plane, out_shape=out_shape, hf_token=opts.hf_token, **opts.remote_params
        )
        return labels, _meta
    z_axis = 0 if do_3D else None
    labels, _count, _meta = local_seg(
        plane, channel_axis=channel_axis, do_3D=do_3D, z_axis=z_axis,
        **opts.local_params,
    )
    return labels, _meta


def run_counting(
    model: axes.AxisModel,
    crops: list[roi.RoiCrop],
    opts: RunOptions,
    *,
    local_seg=_default_local_segmenter,
    remote_seg=_default_remote_segmenter,
) -> RunResult:
    """Segment and count each ROI separately; assemble a composite RunResult.

    ``local_seg`` / ``remote_seg`` are injectable for testing.
    """
    do_3D = bool(opts.do_3D and model.is_zstack and opts.backend == "local")
    is_3d_out = do_3D
    h, w = model.yx_shape
    composite = (
        np.zeros((model.n_z, h, w), dtype=np.int32)
        if is_3d_out
        else np.zeros((h, w), dtype=np.int32)
    )

    per_roi: list[RoiResult] = []
    id_offset = 0
    last_meta: dict = {}

    for crop in crops:
        sub = axes.crop_to_bbox(model, crop.bbox)

        if opts.mode == DP_COLOC:
            plane_a = axes.channel_plane(sub, opts.chan_a)
            plane_b = axes.channel_plane(sub, opts.chan_b)
            labels_a, last_meta = _segment_plane(
                plane_a, None, opts, sub.yx_shape, do_3D, local_seg, remote_seg
            )
            labels_b, _ = _segment_plane(
                plane_b, None, opts, sub.yx_shape, do_3D, local_seg, remote_seg
            )
            labels = projection.colocalize(
                labels_a, labels_b, min_overlap=opts.coloc_min_overlap
            )
        else:
            if opts.mode == DP_MIN:
                plane = projection.min_projection(sub, opts.chan_a, opts.chan_b)
                channel_axis = None
            else:  # SINGLE
                plane, channel_axis = axes.as_segmentation_input(sub, opts.channel)
            labels, last_meta = _segment_plane(
                plane, channel_axis, opts, sub.yx_shape, do_3D, local_seg, remote_seg
            )

        labels = roi.apply_polygon(labels, crop.polygon_mask)
        labels = _drop_small(labels, opts.min_size_refilter)

        new_offset = roi.paste_into(composite, labels, crop.bbox, id_offset)
        region = (
            composite[:, crop.bbox[0]:crop.bbox[2], crop.bbox[1]:crop.bbox[3]]
            if is_3d_out
            else composite[crop.bbox[0]:crop.bbox[2], crop.bbox[1]:crop.bbox[3]]
        )
        ids = sorted(int(i) for i in np.unique(region[region > id_offset]))
        per_roi.append(
            RoiResult(
                roi_index=crop.index,
                count=len(ids),
                label_ids=ids,
                bbox=crop.bbox,
            )
        )
        id_offset = new_offset

    run_meta = dict(last_meta)
    run_meta["mode"] = opts.mode
    run_meta["n_rois"] = len(crops)
    if opts.mode in (DP_MIN, DP_COLOC):
        run_meta["channels"] = {"a": opts.chan_a, "b": opts.chan_b}

    return RunResult(
        labels=composite,
        per_roi=per_roi,
        backend=opts.backend,
        mode=opts.mode,
        run_meta=run_meta,
        roi_polygons=[c.polygon_mask.shape for c in crops],
    )
