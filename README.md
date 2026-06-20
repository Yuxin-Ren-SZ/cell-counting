# Cell Counting by Threshold and Watershed Segmentation

This project counts fluorescently labeled cells in TIFF microscopy images using a classical threshold and watershed segmentation pipeline.

The project includes two ways to run the same algorithm:

- `main.py`: command-line version
- `tk_app.py`: simple Tkinter desktop GUI

The shared segmentation algorithm is stored in `segmentation.py`, so both versions use the same cell counting logic.

## Files

```text
segmentation.py   Core segmentation functions
main.py           Command-line script
tk_app.py         Tkinter desktop GUI
requirements.txt  Required Python packages
README.md         Project instructions
```

## Method

The algorithm follows these steps:

1. Read a TIFF microscopy image.
2. Automatically select the target color channel:
   - NeuN images use the blue channel.
   - Olig2 images use the green channel.
   - Unknown markers use the brightest RGB channel.
3. Apply Gaussian smoothing to reduce noise.
4. Use Otsu thresholding to separate bright cell signal from background.
5. Remove small noisy objects and fill small holes.
6. Use distance transform and watershed segmentation to separate touching cells.
7. Count each segmented region as one cell.
8. Generate an overlay image with red cell boundaries.

## Installation

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

If your system uses `python3`:

```bash
python3 -m pip install -r requirements.txt
```

Anaconda can also be used:

```bash
conda create -n cellcount python=3.11
conda activate cellcount
python -m pip install -r requirements.txt
```

## Run the Desktop GUI

Run:

```bash
python tk_app.py
```

If using Anaconda on this computer:

```bash
/opt/anaconda3/bin/python tk_app.py
```

A desktop window will open. Click:

```text
Choose TIFF Image
```

Select a `.tif` or `.tiff` microscopy image, then click:

```text
Run Segmentation
```

The GUI will display:

- the detected cell count
- the automatically detected channel
- an overlay preview with red segmentation boundaries

It also saves an overlay image next to the input image:

```text
image_name_tk_overlay.png
```

## Run from Command Line

Put the TIFF image in the same folder, then run:

```bash
python main.py "image_name.tif"
```

Example:

```bash
python main.py "9794 NeuN.tif"
```

If using Anaconda:

```bash
/opt/anaconda3/bin/python main.py "9794 NeuN.tif"
```

The command-line version prints the selected channel and cell count:

```text
Using channel: blue
Cell count: 616
```

It also saves an overlay image:

```text
image_name_overlay.png
```

## Parameters

Default parameters are selected based on the image filename:

| Marker | Channel | Threshold Factor | Minimum Object Size | Watershed Min Distance |
|---|---|---:|---:|---:|
| NeuN | Blue | 0.85 | 45 | 7 |
| Olig2 | Green | 0.95 | 20 | 10 |
| Unknown | Brightest RGB channel | 1.00 | 30 | 8 |

These defaults are defined in `segmentation.py`.

## Output

The output overlay image shows the original microscopy image with red segmentation boundaries drawn on top. This is used for visual validation of the segmentation and cell count.

## Notes

This is a classical segmentation baseline. It works best when cells are brighter than the background and have reasonably separated shapes. Dense, blurry, or low-contrast images may require parameter tuning or a more advanced model such as Cellpose.
