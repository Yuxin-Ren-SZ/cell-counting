from pathlib import Path
import sys

import matplotlib.pyplot as plt
import tifffile

from core.baseline import make_overlay, segment_cells


if len(sys.argv) < 2:
    print("Usage: python main.py image.tif")
    sys.exit()

image_path = sys.argv[1]
image = tifffile.imread(image_path)

labels, count, used_channel = segment_cells(image, image_path)

print("Using channel:", used_channel)
print("Cell count:", count)

display, overlay = make_overlay(image, labels)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(display)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title(f"Detected cells: {count}")
plt.axis("off")

plt.tight_layout()
plt.savefig(Path(image_path).stem + "_overlay.png", dpi=200)
plt.close()
