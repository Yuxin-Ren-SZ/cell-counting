"""Reproducibility export bundle, sourced from a :class:`RunResult`.

Adapted from the old ``make_export_widget`` (``napari_gui.py:205-267``) but
driven by the widget's ``RunResult`` instead of a module global, with a
``roi_index`` column in ``counts.csv`` and ``backend``/``mode``/``do_3D`` plus
per-ROI counts recorded in ``params.json``. The overlay is regenerated here
(from the labels) rather than copied from a segmentation side-effect file.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .counting import RunResult


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label_to_roi_map(result: RunResult) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for r in result.per_roi:
        for lid in r.label_ids:
            mapping[int(lid)] = r.roi_index
    return mapping


def _write_overlay(labels: np.ndarray, out_path: Path) -> None:
    """Best-effort colored label overlay PNG (max-projected if 3D)."""
    try:
        from skimage.color import label2rgb
        from skimage.io import imsave

        flat = labels.max(axis=0) if labels.ndim == 3 else labels
        rgb = (label2rgb(flat, bg_label=0) * 255).astype(np.uint8)
        imsave(out_path, rgb, check_contrast=False)
    except Exception as exc:  # noqa: BLE001 - overlay is a nicety, not critical
        print(f"Could not write overlay: {exc}")


def write_export_bundle(
    result: RunResult, out_dir: Path | None = None, timestamp: str | None = None
) -> Path:
    """Write the bundle and return the output directory."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    src = Path(result.image_path) if result.image_path else None

    if out_dir is None:
        base = src.with_name(f"{src.stem}_export_{stamp}") if src else Path(
            f"cell_counter_export_{stamp}"
        )
        out_dir = base
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(result.labels)

    import tifffile

    tifffile.imwrite(out_dir / "labels.tif", labels.astype(np.int32))
    _write_overlay(labels, out_dir / "overlay.png")

    # Per-cell measurements with ROI attribution.
    if labels.max() > 0:
        from skimage.measure import regionprops_table

        props = regionprops_table(labels, properties=("label", "area", "centroid"))
        roi_of = _label_to_roi_map(result)
        is3d = labels.ndim == 3
        if is3d:
            header = "label,roi_index,area_px,centroid_z,centroid_y,centroid_x"
            rows = zip(
                props["label"], props["area"],
                props["centroid-0"], props["centroid-1"], props["centroid-2"],
            )
            lines = [header] + [
                f"{int(l)},{roi_of.get(int(l), -1)},{int(a)},{c0:.2f},{c1:.2f},{c2:.2f}"
                for l, a, c0, c1, c2 in rows
            ]
        else:
            header = "label,roi_index,area_px,centroid_y,centroid_x"
            rows = zip(
                props["label"], props["area"],
                props["centroid-0"], props["centroid-1"],
            )
            lines = [header] + [
                f"{int(l)},{roi_of.get(int(l), -1)},{int(a)},{cy:.2f},{cx:.2f}"
                for l, a, cy, cx in rows
            ]
        (out_dir / "counts.csv").write_text("\n".join(lines) + "\n")

    params_record = {
        **result.run_meta,
        "backend": result.backend,
        "mode": result.mode,
        "total_count": result.total,
        "per_roi": [
            {"roi_index": r.roi_index, "count": r.count, "bbox": list(r.bbox)}
            for r in result.per_roi
        ],
        "input_path": str(src) if src else None,
        "input_sha256": _sha256(src) if src and src.exists() else None,
        "exported_at": stamp,
    }
    (out_dir / "params.json").write_text(json.dumps(params_record, indent=2))
    return out_dir
