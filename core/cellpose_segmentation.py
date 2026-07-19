from pathlib import Path
import numpy as np
from cellpose import models
from cellpose.plot import mask_overlay
import matplotlib.pyplot as plt


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


def segment_cells_cellpose(
    image,
    image_path,
    channel=None,
    diameter=None,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
):
    """Segment with Cellpose-SAM.

    channel=None  -> feed all channels (channel_axis=-1)
    channel=int   -> segment on that single 2D channel only
    diameter=None -> auto-estimate; otherwise expected cell diameter in px
    """
    model = models.CellposeModel(
        gpu=False,
        pretrained_model="cpsam_v2",
    )

    if channel is None:
        seg_input = image
        eval_kwargs = dict(channel_axis=-1)
        used_channel = "CellPoseSAM (cpsam_v2, all channels)"
    else:
        seg_input = extract_channel(image, channel)
        eval_kwargs = dict()
        used_channel = f"CellPoseSAM (cpsam_v2, channel {channel})"

    masks, flows, styles = model.eval(
        seg_input,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        **eval_kwargs,
    )
    count = int(masks.max())

    overlay = mask_overlay(seg_input, masks)
    output_path = Path(image_path).with_name(
        Path(image_path).stem + "_cellpose_overlay.png"
    )
    plt.imsave(output_path, overlay)

    print(f"CellPoseSAM overlay saved to: {output_path}")
    print(f"Channel: {used_channel} | diameter: {diameter} | "
          f"cellprob: {cellprob_threshold} | flow: {flow_threshold}")
    print(f"Cell count: {count}")

    return masks, count, used_channel
