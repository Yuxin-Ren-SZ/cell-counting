"""The Cell Counter dock widget (magicgui-based).

Replaces the old ``napari_gui.py`` widgets and its module globals. The widget
holds all run state on the instance (no globals). It loads a source image into
a canonical :class:`AxisModel` (so every channel / z-slice is available for
channel selection, 3D, and double-positive counting -- a single channel-split
display layer could not provide that), lets the user draw ROIs, runs the
crop-first per-ROI pipeline on the local or remote backend, and exports a
reproducibility bundle.
"""
from __future__ import annotations

from pathlib import Path

import napari
import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSpinBox,
    Label,
    LineEdit,
    PushButton,
    SpinBox,
    Table,
)

from . import pipeline
from ._reader import CHANNEL_CMAPS, load_axis_model
from .axes import to_canonical
from .export import write_export_bundle
from .roi import crops_from_shapes, full_frame_crop

_BACKENDS = [("Local (download model)", "local"), ("Remote (HuggingFace)", "remote")]
_MODES = [
    ("Single channel / all channels", pipeline.SINGLE),
    ("Double-positive: min-projection", pipeline.DP_MIN),
    ("Double-positive: colocalization", pipeline.DP_COLOC),
]
# Local-only eval params disabled in remote mode. (min_size / max_size_fraction
# stay enabled: they're applied as post-filters to the Space's raw masks.)
_LOCAL_ONLY = [
    "diameter", "normalize", "niter", "resample", "augment",
    "tile_overlap", "batch_size", "stitch_threshold",
]


class CellCounterWidget(Container):
    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer
        self._model = None
        self.last_result = None

        # --- source loading ---------------------------------------------
        self._file = FileEdit(label="Image", filter="*.tif *.tiff *.nd2 *")
        self._axis_order = LineEdit(
            label="Axis order (optional)",
            tooltip="Override, e.g. ZYX / CZYX. Leave blank to auto-detect.",
        )
        self._load_btn = PushButton(text="Load image")
        self._load_btn.changed.connect(self._on_load)

        # --- backend / mode ---------------------------------------------
        self._backend = ComboBox(label="Backend", choices=_BACKENDS, value="local")
        self._hf_token = LineEdit(
            label="HF token (optional)",
            tooltip="Leave blank to use a `huggingface-cli login` session or "
            "the HF_TOKEN environment variable. Type a token here to override.",
        )
        self._mode = ComboBox(label="Mode", choices=_MODES, value=pipeline.SINGLE)
        self._channel = ComboBox(label="Channel", choices=[("All channels", -1)],
                                 value=-1)
        self._chan_a = ComboBox(label="Channel A", choices=[("Ch 0", 0)], value=0)
        self._chan_b = ComboBox(label="Channel B", choices=[("Ch 0", 0)], value=0)
        self._coloc = FloatSpinBox(label="Coloc min overlap", value=0.5,
                                   min=0.0, max=1.0, step=0.05)
        self._do_3d = CheckBox(label="3D (z-stack)", value=False)

        for w in (self._backend, self._mode, self._do_3d):
            w.changed.connect(self._refresh_enabled)

        # --- local eval params ------------------------------------------
        self._p = {
            "diameter": FloatSpinBox(label="Diameter (0=auto)", value=0.0,
                                     min=0, max=500, step=1),
            "cellprob_threshold": FloatSpinBox(label="Cell prob thresh", value=0.0,
                                                min=-6.0, max=6.0, step=0.5),
            "flow_threshold": FloatSpinBox(label="Flow thresh", value=0.4,
                                           min=0.0, max=3.0, step=0.1),
            "normalize": CheckBox(label="Normalize", value=True),
            "min_size": SpinBox(label="Min size (px)", value=15, min=-1, max=100000),
            "niter": SpinBox(label="N iterations (0=auto)", value=0, min=0, max=5000,
                             step=10),
            "resample": CheckBox(label="Resample", value=True),
            "augment": CheckBox(label="Augment (TTA)", value=False),
            "max_size_fraction": FloatSpinBox(label="Max size fraction", value=0.4,
                                              min=0.0, max=1.0, step=0.05),
            "tile_overlap": FloatSpinBox(label="Tile overlap", value=0.1,
                                         min=0.0, max=0.5, step=0.05),
            "batch_size": SpinBox(label="Batch size", value=8, min=1, max=64),
            "stitch_threshold": FloatSpinBox(label="Stitch thresh (2.5D)", value=0.0,
                                             min=0.0, max=1.0, step=0.05),
            "anisotropy": FloatSpinBox(label="Anisotropy (3D)", value=1.0,
                                       min=0.0, max=20.0, step=0.5),
        }
        # --- remote params ----------------------------------------------
        # Integers: the HF Space raises a TypeError if these arrive as floats.
        self._resize = SpinBox(label="Max resize", value=1000, min=64,
                               max=8000, step=64)
        self._max_iter = SpinBox(label="Max iterations", value=250, min=1,
                                 max=5000, step=10)

        # --- run / results / export -------------------------------------
        self._run_btn = PushButton(text="Run segmentation")
        self._run_btn.changed.connect(self._on_run)
        self._status = Label(value="Load an image to begin.")
        self._table = Table(value={"data": [], "columns": ["ROI", "Cells"]})
        self._export_btn = PushButton(text="Export results")
        self._export_btn.changed.connect(self._on_export)

        self.extend([
            self._file, self._axis_order, self._load_btn,
            self._backend, self._hf_token,
            self._mode, self._channel, self._chan_a, self._chan_b, self._coloc,
            self._do_3d,
            *self._p.values(),
            self._resize, self._max_iter,
            self._run_btn, self._status, self._table, self._export_btn,
        ])
        self._refresh_enabled()

    # -------------------------------------------------------------------
    def _on_load(self):
        path = str(self._file.value) if self._file.value else ""
        if not path:
            self._status.value = "Pick a file first."
            return
        try:
            order = self._axis_order.value.strip() or None
            if order:
                import tifffile

                data = np.asarray(tifffile.imread(path))
                self._model = to_canonical(data, axis_order=order)
            else:
                self._model = load_axis_model(path)
        except Exception as exc:  # noqa: BLE001
            self._status.value = f"Load failed: {exc}"
            return

        self._image_path = path
        self._add_display_layers()
        self._ensure_roi_layer()
        self._rebuild_channel_choices()
        self._refresh_enabled()
        m = self._model
        self._status.value = (
            f"Loaded {Path(path).name}: {m.n_channels} channel(s), "
            f"{m.n_z} z-slice(s)."
        )

    def _add_display_layers(self):
        m = self._model
        for c in range(m.n_channels):
            chan = m.data[c] if m.is_zstack else m.data[c, 0]
            cmap = CHANNEL_CMAPS[c % len(CHANNEL_CMAPS)]
            self._viewer.add_image(
                chan, name=f"{m.channel_names[c]} ({cmap})",
                colormap=cmap, blending="additive",
            )

    def _ensure_roi_layer(self):
        import napari

        for layer in self._viewer.layers:
            if layer.name == "ROI" and isinstance(layer, napari.layers.Shapes):
                return
        self._viewer.add_shapes(name="ROI", face_color="transparent",
                                edge_color="red", opacity=0.5)

    def _rebuild_channel_choices(self):
        m = self._model
        named = [(f"Ch {i}: {m.channel_names[i]}", i) for i in range(m.n_channels)]
        self._channel.choices = [("All channels", -1)] + named
        self._channel.value = -1
        for combo in (self._chan_a, self._chan_b):
            combo.choices = named
        self._chan_a.value = 0
        self._chan_b.value = min(1, m.n_channels - 1)

    # -------------------------------------------------------------------
    def _refresh_enabled(self):
        remote = self._backend.value == "remote"
        mode = self._mode.value
        is_dp = mode in (pipeline.DP_MIN, pipeline.DP_COLOC)
        zstack = bool(self._model and self._model.is_zstack)

        self._hf_token.enabled = remote
        self._channel.enabled = not is_dp
        self._chan_a.enabled = is_dp
        self._chan_b.enabled = is_dp
        self._coloc.enabled = mode == pipeline.DP_COLOC
        self._do_3d.enabled = zstack and not remote
        if remote or not zstack:
            self._do_3d.value = False

        for name, w in self._p.items():
            w.enabled = not remote or name not in _LOCAL_ONLY
        self._resize.enabled = remote
        self._max_iter.enabled = remote

    # -------------------------------------------------------------------
    def _collect_options(self) -> pipeline.RunOptions:
        p = {k: w.value for k, w in self._p.items()}
        local_params = {
            "diameter": None if p["diameter"] == 0 else p["diameter"],
            "cellprob_threshold": p["cellprob_threshold"],
            "flow_threshold": p["flow_threshold"],
            "normalize": p["normalize"],
            "min_size": p["min_size"],
            "niter": None if p["niter"] == 0 else p["niter"],
            "resample": p["resample"],
            "augment": p["augment"],
            "max_size_fraction": p["max_size_fraction"],
            "tile_overlap": p["tile_overlap"],
            "batch_size": p["batch_size"],
            "stitch_threshold": p["stitch_threshold"],
        }
        if self._do_3d.value:
            local_params["anisotropy"] = (
                None if p["anisotropy"] == 0 else p["anisotropy"]
            )
        remote_params = {
            "resize": self._resize.value,
            "max_iter": self._max_iter.value,
            "flow_threshold": p["flow_threshold"],
            "cellprob_threshold": p["cellprob_threshold"],
            "min_size": p["min_size"],
            "max_size_fraction": p["max_size_fraction"],
        }
        return pipeline.RunOptions(
            backend=self._backend.value,
            channel=None if self._channel.value == -1 else self._channel.value,
            mode=self._mode.value,
            chan_a=self._chan_a.value,
            chan_b=self._chan_b.value,
            do_3D=self._do_3d.value,
            coloc_min_overlap=self._coloc.value,
            min_size_refilter=max(0, p["min_size"]),
            hf_token=self._hf_token.value or None,
            local_params=local_params,
            remote_params=remote_params,
        )

    def _get_crops(self):
        import napari

        yx = self._model.yx_shape
        for layer in self._viewer.layers:
            if layer.name == "ROI" and isinstance(layer, napari.layers.Shapes):
                if len(layer.data) > 0:
                    crops = crops_from_shapes(list(layer.data), yx)
                    if crops:
                        return crops, [np.asarray(s).tolist() for s in layer.data]
        return [full_frame_crop(yx)], []

    def _on_run(self):
        if self._model is None:
            self._status.value = "Load an image first."
            return
        opts = self._collect_options()
        crops, polygons = self._get_crops()
        self._status.value = (
            f"Running {opts.backend} segmentation on {len(crops)} ROI(s)..."
        )
        try:
            result = pipeline.run_counting(self._model, crops, opts)
        except Exception as exc:  # noqa: BLE001
            self._status.value = f"Segmentation failed: {exc}"
            self._notify(f"Cell Counter: {exc}")
            return

        result.image_path = getattr(self, "_image_path", None)
        result.roi_polygons = polygons
        self.last_result = result

        self._viewer.add_labels(
            result.labels, name=f"Cells: {result.total}"
        )
        self._table.value = {
            "data": [[r.roi_index, r.count] for r in result.per_roi]
            + [["TOTAL", result.total]],
            "columns": ["ROI", "Cells"],
        }
        self._status.value = (
            f"Done. {result.total} cells across {len(result.per_roi)} ROI(s)."
        )

    def _on_export(self):
        if self.last_result is None:
            self._status.value = "Nothing to export - run segmentation first."
            return
        try:
            out = write_export_bundle(self.last_result)
        except Exception as exc:  # noqa: BLE001
            self._status.value = f"Export failed: {exc}"
            return
        self._status.value = f"Exported to: {out}"

    def _notify(self, msg: str):
        try:
            from napari.utils.notifications import show_error

            show_error(msg)
        except Exception:  # noqa: BLE001
            pass


def make_cell_counter_widget(napari_viewer: napari.Viewer):
    """npe2 widget factory. napari injects the viewer by type annotation.

    The magicgui Container is wrapped in a real ``QScrollArea`` -- magicgui's
    own ``scrollable=True`` does not size correctly inside a napari dock (the
    top of the form gets clipped and no scrollbar appears).
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QScrollArea

    widget = CellCounterWidget(napari_viewer)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setMinimumWidth(320)
    scroll.setWidget(widget.native)
    # Keep a Python reference to the magicgui widget so it isn't GC'd, and
    # expose it for tests / debugging.
    scroll._counter_widget = widget
    return scroll
