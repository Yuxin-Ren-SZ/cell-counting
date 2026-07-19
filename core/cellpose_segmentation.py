import sys
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from cellpose import models
from cellpose.plot import mask_overlay
import matplotlib.pyplot as plt

MODEL_NAME = "cpsam_v2"

# Cellpose eval() params exposed by this pipeline (2D). 3D-only params
# (do_3D, anisotropy, z_axis, flow3D_smooth) are intentionally omitted.
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


def _pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def get_channel_info(image):
    """Return (channel_axis, n_channels) for a 2D or 3D image.

    Handles both (H, W, C) and (C, H, W) layouts by treating the small
    axis (<= 5) as the channel axis. Returns (None, 1) for 2D images.
    """
    if image.ndim == 2:
        return None, 1
    if image.shape[-1] <= 5:
        return image.ndim - 1, image.shape[-1]
    if image.shape[0] <= 5:
        return 0, image.shape[0]
    # Fallback: assume channels-last.
    return image.ndim - 1, image.shape[-1]


def extract_channel(image, channel):
    """Pull a single 2D channel out of a multi-channel image."""
    axis, n_channels = get_channel_info(image)
    if axis is None:
        return image
    if channel < 0 or channel >= n_channels:
        raise ValueError(f"channel {channel} out of range (0..{n_channels - 1})")
    return np.take(image, channel, axis=axis)


def segment_cells_cellpose(image, image_path, channel=None, **eval_params):
    """Segment with Cellpose-SAM.

    channel=None  -> feed all channels (channel_axis=-1)
    channel=int   -> segment on that single 2D channel only

    eval_params override EVAL_PARAM_DEFAULTS and are passed straight to
    model.eval, so the returned run_meta records exactly what ran.

    Returns (masks, count, used_channel, run_meta). run_meta captures the
    resolved params + software versions + model for reproducibility.
    """
    # Resolve params: defaults overridden by whatever the caller passed.
    params = dict(EVAL_PARAM_DEFAULTS)
    params.update({k: v for k, v in eval_params.items() if k in EVAL_PARAM_DEFAULTS})

    model = models.CellposeModel(gpu=False, pretrained_model=MODEL_NAME)

    if channel is None:
        seg_input = image
        channel_kwargs = dict(channel_axis=-1)
        used_channel = f"CellPoseSAM ({MODEL_NAME}, all channels)"
    else:
        seg_input = extract_channel(image, channel)
        channel_kwargs = dict()
        used_channel = f"CellPoseSAM ({MODEL_NAME}, channel {channel})"

    masks, flows, styles = model.eval(seg_input, **params, **channel_kwargs)
    count = int(masks.max())

    overlay = mask_overlay(seg_input, masks)
    output_path = Path(image_path).with_name(
        Path(image_path).stem + "_cellpose_overlay.png"
    )
    plt.imsave(output_path, overlay)

    run_meta = {
        "model": MODEL_NAME,
        "channel": channel,
        "params": params,
        "versions": {
            "cellpose": _pkg_version("cellpose"),
            "napari": _pkg_version("napari"),
            "numpy": _pkg_version("numpy"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    print(f"CellPoseSAM overlay saved to: {output_path}")
    print(f"Channel: {used_channel} | " + " | ".join(
        f"{k}={v}" for k, v in params.items()))
    print(f"Cell count: {count}")

    return masks, count, used_channel, run_meta
