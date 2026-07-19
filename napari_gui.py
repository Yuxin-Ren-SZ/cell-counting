import sys
import tkinter as tk
from tkinter import filedialog
import napari
import tifffile
import numpy as np
from magicgui import magic_factory
from skimage.draw import polygon2mask
from skimage.measure import label as sklabel
from core.cellpose_segmentation import segment_cells_cellpose

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

@magic_factory
def run_segmentation(image_layer: napari.layers.Image):
    global viewer, image_path
    image = image_layer.data

    roi_layer = None
    for layer in viewer.layers:
        if layer.name == 'ROI' and isinstance(layer, napari.layers.Shapes):
            roi_layer = layer
            break
        roi_mask = np.ones(image.shape[:2], dtype=bool)

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
    labels, count, used_channel = segment_cells_cellpose(image, image_path)
    if not np.all(roi_mask):
        labels_filtered = labels.copy()
        labels_filtered[~roi_mask] = 0
        labels_filtered = sklabel(labels_filtered > 0, connectivity=1)
        count = np.max(labels_filtered)
        labels = labels_filtered
        print(f"Filtered by ROI. Cells in ROI: {count}")
    else:
        print(f"Total cells: {count}")

    viewer.add_labels(labels, name=f'Segmentation Result (Cells: {count})')
    print(f"Done! Detected {count} cells.")

seg_widget = run_segmentation(call_button="Run Segmentation")
viewer.window.add_dock_widget(seg_widget)

napari.run()
