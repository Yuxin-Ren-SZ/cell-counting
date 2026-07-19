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

- **CellPoseSAM integration** – Uses the latest `cpsam_v2` model for high-accuracy cell segmentation, especially for clustered or overlapping cells.
- **Interactive Napari GUI** – Browse, zoom, and adjust layer opacity in a professional image viewer.
- **Freehand ROI selection** – Draw polygons, rectangles, or ellipses directly on the image; only cells inside the ROI are counted.
- **Allen brain atlas overlay** – Load and overlay coronal slices of the Allen Mouse Brain Atlas via `brainrender-napari` to localize cells anatomically.
- **One-click segmentation** – Run the entire pipeline—load image → draw ROI → run CellPoseSAM → display results—with a single button.
- **Automatic output saving** – Segmentation overlays are saved as `*_cellpose_overlay.png` next to the input image.
- **ND2 stitching support** – Load Nikon ND2 files containing multiple stage positions and automatically stitch them into a single large image (via `napari-stitcher`).
- **3D volume segmentation** – Process Z-stack images (3D) using the `cellpose-napari` plugin with the **“process stack as 3D”** option
- **Time‑lapse segmentation** – Segment and track cells in time‑series data using the `stitch_threshold` option in `cellpose‑napari`, enabling 2D‑to‑3D stitching across the time axis.

## Installation

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

If you use `python3`:

```bash
python3 -m pip install -r requirements.txt
```

For Anaconda users:

```bash
conda create -n cellseg python=3.10
conda activate cellseg
python -m pip install -r requirements.txt
```

### Additional Packages for Atlas Overlay

```bash
python -m pip install brainrender-napari brainglobe-atlasapi
```

> **Note:** The first time you load the atlas, it will automatically download the Allen Mouse Brain Atlas, which may require several hundred MB of storage. Ensure that you have a stable internet connection.

## Run the Interactive GUI (Napari)

Navigate to the project folder and run:

```bash
python napari_gui.py
```

A file dialog will open. Select a `.tif` or `.tiff` microscopy image.

## GUI Workflow

1. **Image display** – The image appears in the Napari viewer.

2. **Optional ROI** – Click the **Shapes** tool in the toolbar, then draw a polygon, rectangle, or ellipse on the image. If you skip this step, the entire image is used.

3. **Run segmentation** – In the right panel, click **Run Segmentation**.

4. **Results** – A new labels layer appears overlaid on the image, with each cell colored differently. The layer name shows the cell count.

5. **Atlas overlay (optional)** – Go to `Plugins → Brainrender → Manage atlas versions` and load `allen_mouse_25um`. Once loaded, you can:

   - Use the **slider at the bottom** of the Napari window to browse through different coronal slices and adjust the Z-axis position.
   - Click the **cube icon in the bottom-left corner** to toggle between 2D and 3D views of the atlas.
   - Adjust layer opacity in the left panel to better match your image.

6. **Save** – The overlay image is automatically saved as `original_name_cellpose_overlay.png` in the same folder.

## 3D Segmentation via `cellpose-napari`

If you have Z-stack images (3D), you can use the `cellpose-napari` plugin for 3D segmentation:

1. Launch Napari and load your 3D image, such as a `.tif` Z-stack.

2. Go to `Plugins → cellpose-napari` to open the plugin panel.

3. In the panel, check the **“process stack as 3D”** option.

4. Adjust the other parameters, such as diameter and channel, and click **Run**.

5. The plugin will perform 3D segmentation and display the results as a 3D labels layer.

## Time‑lapse Segmentation (via `cellpose‑napari` plugin)

If you have time‑series images (4D: time + Z + Y + X, or simply a stack of 2D frames over time), you can use the `cellpose‑napari` plugin to segment and track cells across time.
1. Load your time‑series image (e.g., a multi‑frame TIFF) into Napari.

2. Open `cellpose‑napari` (`Plugins` → `cellpose‑napari`).
   
3. In the plugin panel, set **`stitch_threshold`** to a value greater than 0 (e.g., `0.5`). Higher values require more overlap between frames to link cells.
   
4. Adjust other parameters (diameter, channel, etc.) and click **“Run”**.

## Stitching ND2 Images (Optional)

If your microscopy images are stored as Nikon ND2 files with multiple fields of view, you can stitch them into a single large image using the `napari-stitcher` plugin.
## Quick Guide

1. **Directly stitch Napari layers:** Use Napari to load, visualize, and preposition the tiles to be stitched.

2. When working with multi-channel data, follow this naming convention:  
   `{tile} :: {channel}`

3. Load either all layers or only a subset of the layers into the plugin.

4. Choose the registration options, including the registration channel, binning, and other settings.

5. **Stitching = registration + fusion**
   - **Registration:** Refines the tile positions and is optional.
   - **Fusion:** Joins the tiles into a single image.

6. The registration result is displayed in the viewer, and the fused channels are added as new layers.

## ROI Behavior

- If an ROI is drawn, **only cells inside the ROI are counted**; pixels outside are set to background (`0`) before re-labeling.
- Multiple shapes are combined using a union to form the ROI.
- If no ROI is drawn, the whole image is processed.

## Run the Command-Line Baseline

The classical threshold-based pipeline is still available for comparison:

```bash
python main.py "image_name.tif"
```

Example:

```bash
python main.py "9794 NeuN.tif"
```

Example output:

```text
Using channel: blue
Cell count: 616
```

An overlay with red boundaries is saved as `image_name_overlay.png`.

## Run the Tkinter Baseline GUI

For parameter tuning of the classical algorithm:

```bash
python tk_app.py
```

Adjust `Threshold factor`, `Min object size`, and `Watershed min distance`, then click **Run Segmentation**.

Results are previewed and saved as `*_tk_overlay.png`.

## Output

- **For CellPoseSAM:** `original_name_cellpose_overlay.png` – Color-coded cell masks overlaid on the original image.
- **For baseline:** `original_name_overlay.png` – Red outlines around detected cells.
- **For Tkinter GUI:** `original_name_tk_overlay.png`.

All outputs are saved in the same directory as the input image.

## Dependencies

- Python ≥ 3.9
- `numpy`, `scipy`, `scikit-image`
- `tifffile`, `matplotlib`
- `cellpose` — automatically installs PyTorch
- `napari` ≥ 0.4.18
- `magicgui`
- `brainglobe-atlasapi`, `brainrender-napari` — required for atlas support
- `tkinter` — used for file dialogs and included with most Python installations
-(Optional) `cellpose-bapari`,`napari-stitcher` for 3D and stitching
A complete list is available in `requirements.txt`.

