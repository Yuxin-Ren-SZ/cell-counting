# test_cellpose.py
import tifffile
from core.cellpose_segmentation import segment_cells_cellpose

image_path = "9794 NeuN.tif"
image = tifffile.imread(image_path)

labels, count, used_channel = segment_cells_cellpose(image, image_path)

print(f"Model: {used_channel}")
print(f"Detected {count} cells")