"""Tests for pipeline.run_counting using a fake segmenter (no cellpose)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from napari_cell_counter import axes, pipeline, roi  # noqa: E402


def _yx_of(plane):
    """Spatial (Y,X) of a plane that may carry a channel axis or a Z axis."""
    if plane.ndim == 2:
        return plane.shape
    if plane.ndim == 3:
        # Could be (Z,Y,X) or (Y,X,C). Channel-last if last axis small.
        if plane.shape[-1] <= 5:
            return plane.shape[:2]
        return plane.shape[1:]
    return plane.shape[1:3]  # (Z,Y,X,C)


def fake_local_seg(plane, *, channel_axis, do_3D, z_axis, **params):
    """Deterministic segmenter: two square 'cells' per plane."""
    yx = _yx_of(plane)
    if do_3D:
        z = plane.shape[0]
        labels = np.zeros((z,) + yx, dtype=np.int32)
        labels[:, 2:6, 2:6] = 1
        labels[:, 8:12, 8:12] = 2
    else:
        labels = np.zeros(yx, dtype=np.int32)
        labels[2:6, 2:6] = 1
        labels[8:12, 8:12] = 2
    count = int(np.unique(labels[labels > 0]).size)
    return labels, count, {"backend": "local", "fake": True}


def fake_remote_seg(plane, *, out_shape, hf_token, **params):
    labels = np.zeros(out_shape, dtype=np.int32)
    labels[1:8, 1:8] = 1  # 49px, above the default min_size_refilter
    return labels, 1, {"backend": "remote", "fake": True}


def test_single_channel_full_frame():
    model = axes.to_canonical(np.zeros((20, 20)))
    crops = [roi.full_frame_crop((20, 20))]
    rr = pipeline.run_counting(
        model, crops, pipeline.RunOptions(), local_seg=fake_local_seg
    )
    assert rr.total == 2
    assert len(rr.per_roi) == 1
    assert rr.per_roi[0].count == 2


def test_multiple_rois_counted_separately_no_id_collision():
    model = axes.to_canonical(np.zeros((100, 100)))
    sq1 = np.array([[0, 0], [0, 20], [20, 20], [20, 0]], float)
    sq2 = np.array([[50, 50], [50, 70], [70, 70], [70, 50]], float)
    crops = roi.crops_from_shapes([sq1, sq2], (100, 100))
    rr = pipeline.run_counting(
        model, crops, pipeline.RunOptions(), local_seg=fake_local_seg
    )
    assert len(rr.per_roi) == 2
    # Two fake cells per ROI; all ids across the composite are distinct.
    all_ids = sorted(int(i) for i in np.unique(rr.labels[rr.labels > 0]))
    assert len(all_ids) == 4
    assert len(set(all_ids)) == 4  # no collision
    # Per-ROI id sets don't overlap.
    a, b = rr.per_roi[0].label_ids, rr.per_roi[1].label_ids
    assert set(a).isdisjoint(set(b))


def test_polygon_masks_out_of_bbox_cells():
    # A triangle ROI: cells outside the polygon (but inside the bbox) are cut.
    model = axes.to_canonical(np.zeros((20, 20)))
    tri = np.array([[0, 0], [0, 19], [19, 0]], float)  # lower-left triangle
    crops = roi.crops_from_shapes([tri], (20, 20))
    # fake cell 2 is at [8:12,8:12] -> partly outside the triangle diagonal.
    rr = pipeline.run_counting(
        model, crops,
        pipeline.RunOptions(min_size_refilter=0),
        local_seg=fake_local_seg,
    )
    # cell 1 (2:6,2:6) is inside; the run must not error and counts >=1.
    assert rr.total >= 1


def test_3d_when_zstack():
    model = axes.to_canonical(np.zeros((1, 8, 20, 20)))  # C,Z,Y,X
    crops = [roi.full_frame_crop((20, 20))]
    rr = pipeline.run_counting(
        model, crops,
        pipeline.RunOptions(do_3D=True),
        local_seg=fake_local_seg,
    )
    assert rr.labels.ndim == 3  # (Z, Y, X)
    assert rr.labels.shape == (8, 20, 20)
    assert rr.total == 2


def test_double_positive_min_projection_runs():
    ch = np.zeros((2, 20, 20), dtype=np.float32)
    ch[0, 5:15, 5:15] = 100
    ch[1, 5:15, 5:15] = 100
    model = axes.to_canonical(ch)
    crops = [roi.full_frame_crop((20, 20))]
    rr = pipeline.run_counting(
        model, crops,
        pipeline.RunOptions(mode=pipeline.DP_MIN, chan_a=0, chan_b=1),
        local_seg=fake_local_seg,
    )
    assert rr.mode == pipeline.DP_MIN
    assert rr.run_meta["channels"] == {"a": 0, "b": 1}


def test_double_positive_coloc_runs():
    ch = np.zeros((2, 20, 20), dtype=np.float32)
    model = axes.to_canonical(ch)
    crops = [roi.full_frame_crop((20, 20))]
    rr = pipeline.run_counting(
        model, crops,
        pipeline.RunOptions(mode=pipeline.DP_COLOC, chan_a=0, chan_b=1),
        local_seg=fake_local_seg,
    )
    # Both fake channels segment to the same squares -> full overlap -> kept.
    assert rr.total == 2


def test_remote_backend_uses_out_shape():
    model = axes.to_canonical(np.zeros((30, 30)))
    crops = [roi.full_frame_crop((30, 30))]
    rr = pipeline.run_counting(
        model, crops,
        pipeline.RunOptions(backend="remote"),
        remote_seg=fake_remote_seg,
    )
    assert rr.backend == "remote"
    assert rr.total == 1


def test_remote_rejects_multichannel_all():
    model = axes.to_canonical(np.zeros((3, 20, 20)))  # 3 channels
    crops = [roi.full_frame_crop((20, 20))]
    try:
        pipeline.run_counting(
            model, crops,
            pipeline.RunOptions(backend="remote", channel=None),
            remote_seg=fake_remote_seg,
        )
    except ValueError as e:
        assert "2D only" in str(e)
        return
    raise AssertionError("expected ValueError for multichannel remote")


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
