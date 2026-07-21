# napari-cell-counter

A [napari](https://napari.org) plugin for **cell segmentation and counting** with
[Cellpose](https://github.com/MouseLand/cellpose) (Cellpose-SAM). It segments per
**ROI** (cropping to each region first for speed), counts each ROI separately, and can
run segmentation either **locally** (downloads the model to your machine) or **remotely**
on the HuggingFace [Cellpose Space](https://huggingface.co/spaces/mouseland/cellpose)
(GPU compute, no local model needed).

## Features

- **Robust input handling** — single-channel 2D, multi-channel 2D, single-channel
  z-stacks, and multi-channel z-stacks all work. Inputs are normalized to a canonical
  `CZYX` layout using file metadata when available; a manual axis-order override is
  provided for ambiguous files.
- **Reads TIFF and Nikon ND2** (`.tif`, `.tiff`, `.nd2`). ND2 named axes and channel
  names are read directly from the file.
- **Crop-to-ROI segmentation** — each ROI is cropped to its bounding box *before*
  segmentation, so Cellpose processes a smaller image (faster).
- **Per-ROI counts** — draw multiple ROIs and each is counted separately; results are
  shown in a table and pasted into one composite labels layer with non-colliding IDs.
- **Local or remote compute** — choose the local Cellpose model download, or offload to
  the HuggingFace Cellpose Space. An optional HF token grants more daily GPU quota.
- **3D segmentation** — for z-stacks, enable true 3D (`do_3D`) with anisotropy control
  (local backend only).
- **Double-positive counting** — count cells positive in channel A *and* B, via either:
  - **Min-projection** — per-pixel minimum of the two channels (keeps signal only where
    both are bright), then segment. Best for co-nuclear stains.
  - **Colocalization** — segment each channel separately and keep cells that overlap.
- **Reproducible export** — a timestamped bundle with `labels.tif`, `overlay.png`,
  per-cell `counts.csv` (with `roi_index`), and `params.json` (all parameters, backend,
  mode, model, input hash, and library versions).

## Installation

```bash
git clone <this repo> && cd cell-counting
python -m pip install -e .            # local backend
python -m pip install -e ".[remote]"  # also enable HuggingFace remote compute
```

Cellpose pulls in PyTorch. The Cellpose-SAM model (`cpsam_v2`) is downloaded on first
local run to `~/.cellpose/models`.

## Usage

1. Launch napari (`napari`), then open **Plugins → Cell Counter**.
2. In the widget, pick an image file (`.tif/.tiff/.nd2`) and click **Load image**. The
   channels are shown as separate colored layers and an empty **ROI** shapes layer is
   added. (If auto-detection of axes is wrong for a bare TIFF, type an override such as
   `ZYX` or `CZYX` in *Axis order* before loading.)
3. *(Optional)* Select the **ROI** layer and draw one or more polygons/rectangles. With
   no ROI, the whole image is treated as a single region.
4. Choose the **backend** (Local or Remote), the **mode** (single channel, or a
   double-positive method), the channel(s), and tune parameters. For z-stacks, tick
   **3D** (local only).
5. Click **Run segmentation**. A composite labels layer appears and the per-ROI counts
   fill the table.
6. Click **Export results** to write the reproducibility bundle next to the input image.

### Local vs. remote

| | Local | Remote (HuggingFace) |
|---|---|---|
| Compute | your CPU/GPU | HuggingFace ZeroGPU |
| Model download | yes (`~/.cellpose`) | none |
| 3D z-stacks | yes | no (2D only) |
| Parameters | full Cellpose eval set | resize, max iter, flow & cellprob thresholds |
| Quota | unlimited | daily GPU quota (more with an HF token) |

> The local model (`cpsam_v2`) and the Space's Cellpose-SAM are not guaranteed identical,
> so counts may differ slightly between backends. The backend is recorded in the export.

## Development / tests

Pure-logic modules (axis model, ROI cropping, projection, per-ROI counting) are unit
tested without napari or cellpose:

```bash
python -m pytest tests/          # or: python tests/test_core_logic.py
```

## Package layout

```
src/napari_cell_counter/
├── napari.yaml        # npe2 manifest (reader + widget)
├── _reader.py         # tif/tiff/nd2 reader + load_axis_model
├── io_nd2.py          # ND2 loading (named axes)
├── axes.py            # canonical CZYX axis model
├── roi.py             # ROI bbox crop + per-ROI reassembly
├── projection.py      # min-projection & colocalization
├── segmentation.py    # local Cellpose backend (cached model, 3D)
├── remote.py          # HuggingFace Space backend (gradio_client)
├── pipeline.py        # crop → segment → per-ROI count orchestration
├── counting.py        # RoiResult / RunResult
├── export.py          # reproducibility bundle
└── _widget.py         # the Cell Counter dock widget
```
