import sys
import tkinter as tk
from tkinter import filedialog
import napari
import tifffile
import numpy as np
from magicgui import magicgui
from skimage.draw import polygon2mask
from core.cellpose_segmentation import segment_cells_cellpose, get_channel_info

root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
    title="Select a TIFF image",
    filetypes=[("TIFF images", "*.tif *.tiff"), ("All files", "*.*")]
)
root.destroy()

if not image_path:
    print("No file selected. Exiting.")
    sys.exit()

viewer = napari.Viewer()
image_data = tifffile.imread(image_path)
viewer.add_image(image_data, name='Original Image')
viewer.add_shapes(name='ROI', face_color='transparent', edge_color='red', opacity=0.5)

# Build channel dropdown from the loaded image. -1 == all channels.
_, n_channels = get_channel_info(image_data)
channel_choices = [("All channels", -1)] + [
    (f"Channel {i}", i) for i in range(n_channels)
]

@magicgui(
    call_button="Run Segmentation",
    channel={"choices": channel_choices, "label": "Channel"},
    diameter={"label": "Diameter (px, 0 = auto)", "min": 0, "max": 500, "step": 1},
    cellprob_threshold={"label": "Cell prob thresh", "min": -6.0, "max": 6.0, "step": 0.5},
    flow_threshold={"label": "Flow thresh", "min": 0.0, "max": 3.0, "step": 0.1},
)
def run_segmentation(
    image_layer: napari.layers.Image,
    channel: int = -1,
    diameter: float = 0.0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
):
    global viewer, image_path
    image = image_layer.data
    channel_arg = None if channel == -1 else channel
    diameter_arg = None if diameter == 0 else diameter

    roi_mask = np.ones(image.shape[:2], dtype=bool)
    roi_layer = None
    for layer in viewer.layers:
        if layer.name == 'ROI' and isinstance(layer, napari.layers.Shapes):
            roi_layer = layer
            break

    if roi_layer is not None and len(roi_layer.data) > 0:
        mask = np.zeros(image.shape[:2], dtype=bool)
        for shape_data in roi_layer.data:
            coords = shape_data
            mask_segment = polygon2mask(image.shape[:2], coords)
            mask = np.logical_or(mask, mask_segment)
        roi_mask = mask
        print("ROI mask generated.")
    else:
        print("No ROI drawn. Using full image.")

    print("Running CellPoseSAM segmentation...")
    labels, count, used_channel = segment_cells_cellpose(
        image,
        image_path,
        channel=channel_arg,
        diameter=diameter_arg,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
    )
    if not np.all(roi_mask):
        # Keep any cell with at least one pixel inside the ROI, counted whole
        # (not sliced) with Cellpose IDs preserved. See note below re: border cells.
        keep_ids = np.unique(labels[roi_mask & (labels > 0)])
        labels_filtered = np.where(np.isin(labels, keep_ids), labels, 0)
        count = int(keep_ids.size)
        labels = labels_filtered
        print(f"Filtered by ROI. Cells in ROI: {count}")
    else:
        count = int(np.unique(labels[labels > 0]).size)
        print(f"Total cells: {count}")

    viewer.add_labels(labels, name=f'Segmentation Result (Cells: {count})')
    print(f"Done! Detected {count} cells.")

viewer.window.add_dock_widget(run_segmentation, name="Cellpose Segmentation")

napari.run()
