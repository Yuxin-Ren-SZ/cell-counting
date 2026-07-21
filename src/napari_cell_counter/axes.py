"""Canonical axis model for cell-counting inputs.

Every input image -- however it arrives (TIFF, ND2, a napari layer, a bare
ndarray) -- is normalized to a canonical ``CZYX`` layout with explicit,
named axes. Channel and Z axes are always present (they may be size 1).

This replaces the old size-``<=5`` axis guesser in
``core/cellpose_segmentation.py`` and, crucially, removes the crash where a
single-channel 2D image segmented with the "All channels" option was fed to
Cellpose with ``channel_axis=-1`` (which misread the image width as channels).
Here, "all channels" on a single-channel image simply yields the one 2D plane
with no channel axis.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Axis letters we understand. T (time) and S (RGB samples) are handled during
# normalization: T is reduced to its first index, S is treated as channels.
_SPATIAL = ("Y", "X")


@dataclass
class AxisModel:
    """An image normalized to canonical ``CZYX`` (channel, z, y, x).

    ``data`` is always 4D. ``n_channels``/``n_z`` may be 1. ``channel_names``
    has length ``n_channels``. ``source_order`` records the axis order the data
    arrived in (for provenance / debugging).
    """

    data: np.ndarray
    channel_names: list[str]
    source_order: str

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_z(self) -> int:
        return self.data.shape[1]

    @property
    def yx_shape(self) -> tuple[int, int]:
        return self.data.shape[2], self.data.shape[3]

    @property
    def is_zstack(self) -> bool:
        return self.n_z > 1


def infer_axis_order(shape: tuple[int, ...]) -> str:
    """Best-effort axis order for a bare ndarray with no metadata.

    Heuristic (advisory only -- the UI can always override):
      2D            -> YX
      3D, small last-> YXC        (channels-last, e.g. HxWx3)
      3D, small 1st -> CYX        (channels-first)
      3D otherwise  -> ZYX        (single-channel z-stack)
      4D            -> CZYX / ZCYX (whichever leading axis looks like channels)
    "small" means size <= 5.
    """
    nd = len(shape)
    if nd == 2:
        return "YX"
    if nd == 3:
        if shape[-1] <= 5:
            return "YXC"
        if shape[0] <= 5:
            return "CYX"
        return "ZYX"
    if nd == 4:
        a, b = shape[0], shape[1]
        # Channels are usually the smaller of the two leading axes.
        if b <= 5 and (a > 5 or b < a):
            return "ZCYX"
        return "CZYX"
    raise ValueError(f"Unsupported image with {nd} dimensions: shape={shape}")


def _to_czyx(data: np.ndarray, order: str) -> np.ndarray:
    """Permute ``data`` (whose axes are described by ``order``) to CZYX."""
    order = order.upper().replace("S", "C")
    if len(order) != data.ndim:
        raise ValueError(
            f"axis order '{order}' has {len(order)} axes but data is "
            f"{data.ndim}-dimensional (shape {data.shape})"
        )
    if "Y" not in order or "X" not in order:
        raise ValueError(f"axis order '{order}' must contain Y and X")

    # Reduce any axis that is not one of C/Z/Y/X (e.g. T) to its first index.
    slicer: list = [slice(None)] * data.ndim
    reduced_order = ""
    for ax, ch in enumerate(order):
        if ch in "CZYX":
            reduced_order += ch
        else:
            slicer[ax] = 0
    data = data[tuple(slicer)]

    # Insert missing channel / z axes as size-1 leading dims.
    for ch in "CZ":
        if ch not in reduced_order:
            data = data[np.newaxis, ...]
            reduced_order = ch + reduced_order

    pos = {ch: i for i, ch in enumerate(reduced_order)}
    perm = [pos[c] for c in "CZYX"]
    return np.transpose(data, perm)


def to_canonical(
    data: np.ndarray,
    axis_order: str | None = None,
    channel_names: list[str] | None = None,
) -> AxisModel:
    """Normalize any array into an :class:`AxisModel` (canonical CZYX).

    ``axis_order`` (e.g. ``"TZCYX"`` from ND2 metadata) is used verbatim when
    given -- we never guess if we know. Otherwise :func:`infer_axis_order` is
    consulted; the result is always overridable by passing ``axis_order``.
    """
    data = np.asarray(data)
    order = axis_order if axis_order else infer_axis_order(data.shape)
    czyx = _to_czyx(data, order)

    n_channels = czyx.shape[0]
    if channel_names is not None:
        names = [str(n) for n in channel_names[:n_channels]]
        # Pad if metadata gave fewer names than channels.
        names += [f"Channel {i}" for i in range(len(names), n_channels)]
    else:
        names = [f"Channel {i}" for i in range(n_channels)]

    return AxisModel(data=czyx, channel_names=names, source_order=order)


def crop_to_bbox(model: AxisModel, bbox: tuple[int, int, int, int]) -> AxisModel:
    """Return a new model cropped in Y/X to ``bbox`` = (y0, x0, y1, x1).

    Channel and Z axes are kept whole, so the array fed to segmentation is
    smaller only in the spatial dimensions.
    """
    y0, x0, y1, x1 = bbox
    cropped = model.data[:, :, y0:y1, x0:x1]
    return replace(model, data=cropped)


def channel_plane(model: AxisModel, channel: int) -> np.ndarray:
    """Return a single channel as ``(Y,X)`` (2D) or ``(Z,Y,X)`` (z-stack)."""
    if channel < 0 or channel >= model.n_channels:
        raise ValueError(
            f"channel {channel} out of range (0..{model.n_channels - 1})"
        )
    plane = model.data[channel]  # (Z, Y, X)
    if not model.is_zstack:
        return plane[0]  # (Y, X)
    return plane


def as_segmentation_input(
    model: AxisModel, channel: int | None
) -> tuple[np.ndarray, int | None]:
    """Prepare the array + channel-axis to hand to Cellpose.

    ``channel=int``  -> that single channel, ``channel_axis=None``.
    ``channel=None`` ("all channels"):
        * single-channel image -> the one 2D/3D plane, ``channel_axis=None``
          (this is the fix for the historical single-channel crash);
        * multi-channel image  -> channels moved to the last axis, i.e.
          ``(Y,X,C)`` or ``(Z,Y,X,C)``, with ``channel_axis=-1``.
    """
    if channel is not None:
        return channel_plane(model, channel), None

    if model.n_channels == 1:
        return channel_plane(model, 0), None

    # Multi-channel "all channels": move C to the last axis, drop singleton Z.
    data = model.data  # (C, Z, Y, X)
    if model.is_zstack:
        arr = np.moveaxis(data, 0, -1)  # (Z, Y, X, C)
    else:
        arr = np.moveaxis(data[:, 0], 0, -1)  # (Y, X, C)
    return arr, -1
