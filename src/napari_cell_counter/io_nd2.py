"""Nikon ND2 loading.

Uses the ``nd2`` package, which exposes named axes via ``ND2File.sizes`` (an
ordered mapping like ``{'T':1,'Z':12,'C':2,'Y':2048,'X':2048}``) that maps
directly to an axis-order string -- so this path never guesses. Real stain
names come from the ND2 channel metadata.
"""
from __future__ import annotations

import numpy as np


def read_nd2(path) -> tuple[np.ndarray, str, list[str] | None]:
    """Return ``(data, axis_order, channel_names)`` for an ND2 file.

    ``axis_order`` is a string of the array's axes (e.g. ``"TZCYX"``); it is
    fed verbatim to :func:`axes.to_canonical`, which reduces T and normalizes
    to CZYX.
    """
    import nd2

    with nd2.ND2File(str(path)) as f:
        sizes = f.sizes  # ordered dict: axis letter -> length
        order = "".join(sizes.keys())
        data = f.asarray()  # numpy array in the order of sizes.keys()
        names = None
        try:
            names = [c.channel.name for c in f.metadata.channels]
        except Exception:  # noqa: BLE001 - channel metadata is optional
            names = None
    return np.asarray(data), order, names
