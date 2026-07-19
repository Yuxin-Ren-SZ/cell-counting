from pathlib import Path
import numpy as np
from cellpose import models
from cellpose.plot import mask_overlay
import matplotlib.pyplot as plt

def segment_cells_cellpose(image, image_path, diameter=None):
    model = models.CellposeModel(
    gpu=False,
    pretrained_model="cpsam_v2",
)
    masks, flows, styles = model.eval(
    image,
    channel_axis=-1,
    diameter=diameter,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
)
    count = int(masks.max())

    overlay = mask_overlay(image, masks)
    output_path = Path(image_path).with_name(
        Path(image_path).stem + "_cellpose_overlay.png"
)
    
    plt.imsave(output_path, overlay)
    
    print(f"CellPoseSAM overlay saved to: {output_path}")
    print(f"Cell count: {count}")
    
    used_channel = "CellPoseSAM (cpsam_v2, auto-channel)"
    return masks, count, used_channel
