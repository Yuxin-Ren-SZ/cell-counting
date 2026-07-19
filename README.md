# Cell Segmentation with CellPoseSAM and Interactive Atlas Overlay

This project extends a classical cell-counting pipeline by integrating **CellPoseSAM** (a deep learning model) into an interactive **Napari**-based GUI. It supports freehand ROI selection, semi-automatic Allen mouse brain atlas overlay, and one-click segmentation with instant visual feedback.

The project provides three main interfaces:

- `napari_gui.py`: **primary GUI** with ROI drawing, CellPoseSAM segmentation, and atlas overlay (recommended)
- `main.py`: command-line baseline (threshold + watershed)
- `tk_app.py`: Tkinter GUI for baseline parameter tuning

All segmentation cores are stored in `core/`, allowing the same logic to be reused across interfaces.

## Files

```text
core/
├── baseline.py                 # Otsu threshold + watershed (baseline)
└── cellpose_segmentation.py    # CellPoseSAM (cpsam_v2) wrapper

napari_gui.py                   # Main Napari-based interactive GUI
main.py                         # Command-line baseline
tk_app.py                       # Tkinter baseline GUI
requirements.txt                # Python dependencies
README.md                       # This file
```
## Features

CellPoseSAM integration – Uses the latest cpsam_v2 model for high‑accuracy cell segmentation, especially for clustered or overlapping cells.
Interactive Napari GUI – Browse, zoom, and adjust layer opacity in a professional image viewer.
Freehand ROI selection – Draw polygons, rectangles, or ellipses directly on the image; only cells inside the ROI are counted.
Allen brain atlas overlay – Load and overlay coronal slices of the Allen Mouse Brain Atlas (via brainrender-napari) to localize cells anatomically.
One‑click segmentation – Run the entire pipeline (load image → draw ROI → run CellPoseSAM → display results) with a single button.
Automatic output saving – Segmentation overlays are saved as *_cellpose_overlay.png next to the input image.
