"""napari reader contribution + shared image loading.

Registers a reader for ``.tif/.tiff/.nd2``. On its own it loads the file into
napari as (channel-split) image layers, and it also exposes
:func:`load_axis_model`, which the Cell Counter widget uses to obtain a
canonical :class:`~napari_cell_counter.axes.AxisModel` from a path.
"""
from __future__ import annotations

import re

import numpy as np

from .axes import AxisModel, to_canonical

CHANNEL_CMAPS = ["red", "green", "blue", "magenta", "cyan", "yellow"]
_TIFF_EXTS = (".tif", ".tiff")
_ND2_EXT = ".nd2"


def read_channel_names(path, n_channels: int) -> list[str] | None:
    """Best-effort channel names from TIFF metadata (ImageJ, then OME-XML).

    Moved from the old ``napari_gui.py``; napari cannot infer stain identity
    from pixels, so this surfaces it when the file records it.
    """
    import tifffile

    try:
        with tifffile.TiffFile(path) as tif:
            ij = tif.imagej_metadata or {}
            labels = ij.get("Labels")
            if labels and len(labels) >= n_channels:
                names = [str(x) for x in labels[:n_channels]]
                if len(set(names)) > 1 or n_channels == 1:
                    return names

            ome = getattr(tif, "ome_metadata", None)
            if ome:
                found = re.findall(r'<Channel[^>]*\bName="([^"]+)"', ome)
                if len(found) >= n_channels:
                    return found[:n_channels]
    except Exception as exc:  # noqa: BLE001 - metadata is optional
        print(f"Could not read channel names: {exc}")
    return None


def _read_raw(path) -> tuple[np.ndarray, str | None, list[str] | None]:
    """Return ``(data, axis_order_or_None, channel_names_or_None)``."""
    p = str(path).lower()
    if p.endswith(_ND2_EXT):
        from .io_nd2 import read_nd2

        return read_nd2(path)
    if p.endswith(_TIFF_EXTS):
        import tifffile

        data = np.asarray(tifffile.imread(path))
        return data, None, None
    raise ValueError(f"unsupported file type: {path}")


def load_axis_model(path) -> AxisModel:
    """Load any supported file into a canonical :class:`AxisModel`.

    For TIFFs, channel names are probed from metadata *after* the axis model is
    built (so we know how many channels to look for).
    """
    data, order, names = _read_raw(path)
    model = to_canonical(data, axis_order=order, channel_names=names)
    if names is None and str(path).lower().endswith(_TIFF_EXTS):
        tiff_names = read_channel_names(path, model.n_channels)
        if tiff_names:
            model = to_canonical(data, axis_order=order, channel_names=tiff_names)
    return model


def _layer_data(path):
    """Build napari layer tuples: one colored image layer per channel."""
    model = load_axis_model(path)
    layers = []
    for c in range(model.n_channels):
        chan = model.data[c]  # (Z, Y, X)
        if not model.is_zstack:
            chan = chan[0]  # (Y, X)
        cmap = CHANNEL_CMAPS[c % len(CHANNEL_CMAPS)]
        name = f"{model.channel_names[c]} ({cmap})"
        layers.append(
            (chan, {"name": name, "colormap": cmap, "blending": "additive"}, "image")
        )
    return layers


def napari_get_reader(path):
    """npe2 reader hook. Returns a reader callable for supported files."""
    if isinstance(path, list):
        path = path[0] if path else ""
    p = str(path).lower()
    if p.endswith(_TIFF_EXTS) or p.endswith(_ND2_EXT):
        return lambda pth: _layer_data(pth if not isinstance(pth, list) else pth[0])
    return None
