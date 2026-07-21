"""Local Cellpose segmentation backend (Cellpose-SAM ``cpsam_v2``).

Changes from the old ``core/cellpose_segmentation.py``:

* the model is built **once and cached** (the old code rebuilt ``CellposeModel``
  -- reloading weights -- on every run);
* **no disk side-effects** in the hot path (the old code wrote an overlay PNG
  every call); overlays are now the export step's job;
* operates on an **already channel-selected / projected** 2D or 3D array, so
  channel handling lives entirely in :mod:`axes` -- this is what removes the
  "All channels + single-channel image" crash class;
* exposes Cellpose's true **3D** params (``do_3D``, ``anisotropy``, ``z_axis``,
  ``flow3D_smooth``) for z-stacks.
"""
from __future__ import annotations

import platform
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import numpy as np

MODEL_NAME = "cpsam_v2"

# 2D eval() params exposed by the GUI (unchanged from the original pipeline).
EVAL_PARAM_DEFAULTS = {
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "normalize": True,
    "min_size": 15,
    "niter": None,
    "resample": True,
    "augment": False,
    "max_size_fraction": 0.4,
    "tile_overlap": 0.1,
    "batch_size": 8,
    "stitch_threshold": 0.0,
}

# 3D-only params, forwarded only when do_3D is requested (z-stack inputs).
EVAL_PARAM_DEFAULTS_3D = {
    "anisotropy": None,
    "flow3D_smooth": 0.0,
}


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


@lru_cache(maxsize=2)
def get_model(gpu: bool = False, model_name: str = MODEL_NAME):
    """Return a cached ``CellposeModel``. Imported lazily so this module can be
    imported without cellpose installed (e.g. for the remote-only path)."""
    from cellpose import models

    return models.CellposeModel(gpu=gpu, pretrained_model=model_name)


def segment_local(
    plane: np.ndarray,
    *,
    channel_axis: int | None = None,
    do_3D: bool = False,
    z_axis: int | None = None,
    gpu: bool = False,
    **eval_params,
) -> tuple[np.ndarray, int, dict]:
    """Segment a prepared ``(Y,X)`` / ``(Z,Y,X)`` (optionally channel-bearing)
    array with local Cellpose. Returns ``(labels, count, run_meta)``.
    """
    params = dict(EVAL_PARAM_DEFAULTS)
    if do_3D:
        params.update(EVAL_PARAM_DEFAULTS_3D)
    params.update({k: v for k, v in eval_params.items() if k in params})

    model = get_model(gpu=gpu, model_name=MODEL_NAME)

    eval_kwargs = dict(params)
    if channel_axis is not None:
        eval_kwargs["channel_axis"] = channel_axis
    if do_3D:
        eval_kwargs["do_3D"] = True
        if z_axis is not None:
            eval_kwargs["z_axis"] = z_axis

    masks, _flows, _styles = model.eval(plane, **eval_kwargs)
    labels = np.asarray(masks)
    count = int(np.unique(labels[labels > 0]).size)

    run_meta = {
        "backend": "local",
        "model": MODEL_NAME,
        "gpu": gpu,
        "do_3D": do_3D,
        "params": params,
        "versions": {
            "cellpose": _pkg_version("cellpose"),
            "napari": _pkg_version("napari"),
            "numpy": _pkg_version("numpy"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    return labels, count, run_meta
