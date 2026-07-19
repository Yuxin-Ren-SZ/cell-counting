import sys
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import napari
import tifffile
import numpy as np
from magicgui import magicgui
from skimage.draw import polygon2mask
from skimage.measure import regionprops_table
from core.cellpose_segmentation import segment_cells_cellpose, get_channel_info

# Distinct colormaps cycled across channels for the split display.
CHANNEL_CMAPS = ["red", "green", "blue", "magenta", "cyan", "yellow"]

# Runtime state, set by main(). run_segmentation/export_results read these.
viewer = None
image_data = None
image_path = None
# Populated after each segmentation run so export reflects the actual run,
# not the current slider values.
last_run = None


def read_channel_names(image_path, n_channels):
    """Best-effort real channel names from TIFF metadata.

    Probes ImageJ metadata, then OME-XML Channel@Name. Returns a list of
    length n_channels, or None if no usable names are found. napari cannot
    infer stain identity from pixels; this surfaces it when the file records it.
    """
    try:
        with tifffile.TiffFile(image_path) as tif:
            ij = tif.imagej_metadata or {}
            # ImageJ stores per-slice labels in 'Labels'; use the first n.
            labels = ij.get("Labels")
            if labels and len(labels) >= n_channels:
                names = [str(x) for x in labels[:n_channels]]
                if len(set(names)) > 1 or n_channels == 1:
                    return names

            ome = getattr(tif, "ome_metadata", None)
            if ome:
                import re
                found = re.findall(r'<Channel[^>]*\bName="([^"]+)"', ome)
                if len(found) >= n_channels:
                    return found[:n_channels]
    except Exception as exc:  # noqa: BLE001 - metadata is optional
        print(f"Could not read channel names: {exc}")
    return None


def add_image_layers(viewer, image_data, image_path):
    """Add the image as per-channel colored layers and print the mapping.

    Multichannel -> one colored layer per channel (additive blend). The
    combined array is not used as a layer, but the channel *indices* still
    match what the segmentation dropdown feeds to Cellpose.
    """
    axis, n_channels = get_channel_info(image_data)
    if axis is None or n_channels == 1:
        viewer.add_image(image_data, name="Channel 0 (gray)", colormap="gray")
        print("Loaded single-channel (grayscale) image.")
        return

    real_names = read_channel_names(image_path, n_channels)
    cmaps = [CHANNEL_CMAPS[i % len(CHANNEL_CMAPS)] for i in range(n_channels)]
    names = []
    print("Channel -> color mapping "
          "(napari cannot infer stain identity from pixels):")
    for i in range(n_channels):
        real = real_names[i] if real_names else None
        label = f"Ch{i} ({cmaps[i]})" + (f" [{real}]" if real else "")
        names.append(label)
        print(f"  Channel {i} -> {cmaps[i]} colormap"
              + (f"  (name: {real})" if real else ""))

    viewer.add_image(
        image_data,
        channel_axis=axis,
        name=names,
        colormap=cmaps,
        blending="additive",
    )


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_segmentation_widget(channel_choices):
    """Build the segmentation dock widget. Segments the full loaded array;
    the Channel dropdown selects which channel Cellpose uses."""

    @magicgui(
        call_button="Run Segmentation",
        channel={"choices": channel_choices, "label": "Channel"},
        diameter={"label": "Diameter (px, 0 = auto)", "min": 0, "max": 500, "step": 1},
        cellprob_threshold={"label": "Cell prob thresh", "min": -6.0, "max": 6.0, "step": 0.5},
        flow_threshold={"label": "Flow thresh", "min": 0.0, "max": 3.0, "step": 0.1},
        normalize={"label": "Normalize"},
        min_size={"label": "Min size (px)", "min": -1, "max": 100000, "step": 1},
        niter={"label": "N iterations (0 = auto)", "min": 0, "max": 5000, "step": 10},
        resample={"label": "Resample"},
        augment={"label": "Augment (TTA)"},
        max_size_fraction={"label": "Max size fraction", "min": 0.0, "max": 1.0, "step": 0.05},
        tile_overlap={"label": "Tile overlap", "min": 0.0, "max": 0.5, "step": 0.05},
        batch_size={"label": "Batch size", "min": 1, "max": 64, "step": 1},
        stitch_threshold={"label": "Stitch thresh (2.5D)", "min": 0.0, "max": 1.0, "step": 0.05},
    )
    def run_segmentation(
        channel: int = -1,
        diameter: float = 0.0,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        normalize: bool = True,
        min_size: int = 15,
        niter: int = 0,
        resample: bool = True,
        augment: bool = False,
        max_size_fraction: float = 0.4,
        tile_overlap: float = 0.1,
        batch_size: int = 8,
        stitch_threshold: float = 0.0,
    ):
        global last_run
        image = image_data
        channel_arg = None if channel == -1 else channel
        diameter_arg = None if diameter == 0 else diameter
        niter_arg = None if niter == 0 else niter

        roi_mask = np.ones(image.shape[:2], dtype=bool)
        roi_coords = []
        roi_layer = None
        for layer in viewer.layers:
            if layer.name == 'ROI' and isinstance(layer, napari.layers.Shapes):
                roi_layer = layer
                break

        if roi_layer is not None and len(roi_layer.data) > 0:
            mask = np.zeros(image.shape[:2], dtype=bool)
            for shape_data in roi_layer.data:
                coords = shape_data
                roi_coords.append(np.asarray(coords).tolist())
                mask_segment = polygon2mask(image.shape[:2], coords)
                mask = np.logical_or(mask, mask_segment)
            roi_mask = mask
            print("ROI mask generated.")
        else:
            print("No ROI drawn. Using full image.")

        print("Running CellPoseSAM segmentation...")
        labels, count, used_channel, run_meta = segment_cells_cellpose(
            image,
            image_path,
            channel=channel_arg,
            diameter=diameter_arg,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            normalize=normalize,
            min_size=min_size,
            niter=niter_arg,
            resample=resample,
            augment=augment,
            max_size_fraction=max_size_fraction,
            tile_overlap=tile_overlap,
            batch_size=batch_size,
            stitch_threshold=stitch_threshold,
        )
        if not np.all(roi_mask):
            # Keep any cell with at least one pixel inside the ROI, counted
            # whole (not sliced) with Cellpose IDs preserved.
            keep_ids = np.unique(labels[roi_mask & (labels > 0)])
            labels_filtered = np.where(np.isin(labels, keep_ids), labels, 0)
            count = int(keep_ids.size)
            labels = labels_filtered
            print(f"Filtered by ROI. Cells in ROI: {count}")
        else:
            count = int(np.unique(labels[labels > 0]).size)
            print(f"Total cells: {count}")

        viewer.add_labels(labels, name=f'Segmentation Result (Cells: {count})')

        last_run = {
            "labels": labels,
            "roi_mask": None if np.all(roi_mask) else roi_mask,
            "roi_coords": roi_coords,
            "count": count,
            "run_meta": run_meta,
            "image_path": image_path,
        }
        print(f"Done! Detected {count} cells.")

    return run_segmentation


def make_export_widget():
    """Build the export dock widget."""

    @magicgui(call_button="Export results")
    def export_results():
        """Write a timestamped reproducibility bundle next to the input image."""
        if last_run is None:
            print("Nothing to export yet - run segmentation first.")
            return

        src = Path(last_run["image_path"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = src.with_name(f"{src.stem}_export_{stamp}")
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = last_run["labels"]
        tifffile.imwrite(out_dir / "labels.tif", labels.astype(np.int32))

        overlay_png = src.with_name(src.stem + "_cellpose_overlay.png")
        if overlay_png.exists():
            shutil.copy(overlay_png, out_dir / "overlay.png")

        if last_run["roi_mask"] is not None:
            np.save(out_dir / "roi.npy", last_run["roi_mask"])
            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"shape_index": i},
                        "geometry": {"type": "Polygon", "coordinates": [coords]},
                    }
                    for i, coords in enumerate(last_run["roi_coords"])
                ],
            }
            (out_dir / "roi.geojson").write_text(json.dumps(geojson, indent=2))

        # Per-cell measurements.
        if labels.max() > 0:
            props = regionprops_table(
                labels, properties=("label", "area", "centroid")
            )
            rows = zip(
                props["label"], props["area"],
                props["centroid-0"], props["centroid-1"],
            )
            lines = ["label,area_px,centroid_y,centroid_x"]
            lines += [f"{int(l)},{int(a)},{cy:.2f},{cx:.2f}"
                      for l, a, cy, cx in rows]
            (out_dir / "counts.csv").write_text("\n".join(lines) + "\n")

        params_record = {
            **last_run["run_meta"],
            "count": last_run["count"],
            "input_path": str(src),
            "input_sha256": _sha256(src),
            "exported_at": stamp,
        }
        (out_dir / "params.json").write_text(json.dumps(params_record, indent=2))

        print(f"Exported reproducibility bundle to: {out_dir}")

    return export_results


def main():
    global viewer, image_data, image_path

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
    add_image_layers(viewer, image_data, image_path)
    viewer.add_shapes(name='ROI', face_color='transparent',
                      edge_color='red', opacity=0.5)

    # Build channel dropdown from the loaded image. -1 == all channels.
    _, n_channels = get_channel_info(image_data)
    channel_choices = [("All channels", -1)] + [
        (f"Channel {i}", i) for i in range(n_channels)
    ]

    viewer.window.add_dock_widget(
        make_segmentation_widget(channel_choices), name="Cellpose Segmentation")
    viewer.window.add_dock_widget(make_export_widget(), name="Export")

    napari.run()


if __name__ == "__main__":
    main()
