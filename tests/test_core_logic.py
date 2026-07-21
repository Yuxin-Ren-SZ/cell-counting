"""Unit tests for the pure-logic modules (no napari / cellpose needed).

Run with the lightweight venv:
    <venv>/bin/python -m pytest tests/test_core_logic.py
or standalone:
    <venv>/bin/python tests/test_core_logic.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from napari_cell_counter import axes, projection, roi, counting  # noqa: E402
from napari_cell_counter.remote import filter_labels  # noqa: E402 (numpy-only)


# --------------------------------------------------------------------------
# axes: the shape matrix required by the plan (#7)
# --------------------------------------------------------------------------
def test_single_channel_2d():
    m = axes.to_canonical(np.zeros((100, 120)))
    assert m.data.shape == (1, 1, 100, 120)
    assert m.n_channels == 1 and m.n_z == 1 and not m.is_zstack


def test_multichannel_channels_last():
    m = axes.to_canonical(np.zeros((100, 120, 3)))
    assert m.data.shape == (3, 1, 100, 120)
    assert m.n_channels == 3 and not m.is_zstack


def test_multichannel_channels_first():
    m = axes.to_canonical(np.zeros((3, 100, 120)))
    assert m.data.shape == (3, 1, 100, 120)


def test_single_channel_zstack():
    m = axes.to_canonical(np.zeros((12, 100, 120)))  # ZYX
    assert m.data.shape == (1, 12, 100, 120)
    assert m.is_zstack and m.n_channels == 1


def test_multichannel_zstack_czyx():
    m = axes.to_canonical(np.zeros((2, 12, 100, 120)))
    assert m.data.shape == (2, 12, 100, 120)
    assert m.n_channels == 2 and m.n_z == 12


def test_multichannel_zstack_zcyx():
    m = axes.to_canonical(np.zeros((12, 2, 100, 120)))  # Z,C,Y,X inferred
    assert m.data.shape == (2, 12, 100, 120)


def test_explicit_order_beats_heuristic():
    # A (4, 100, 120) volume that is really a 4-plane z-stack, not 4 channels.
    m = axes.to_canonical(np.zeros((4, 100, 120)), axis_order="ZYX")
    assert m.data.shape == (1, 4, 100, 120)
    assert m.is_zstack


def test_tczyx_drops_time():
    m = axes.to_canonical(np.zeros((3, 2, 5, 100, 120)), axis_order="TCZYX")
    assert m.data.shape == (2, 5, 100, 120)  # T=0 taken


def test_channel_names_padded():
    m = axes.to_canonical(np.zeros((3, 100, 120)), channel_names=["DAPI"])
    assert m.channel_names == ["DAPI", "Channel 1", "Channel 2"]


# --------------------------------------------------------------------------
# axes: the "All channels" crash fix (#7)
# --------------------------------------------------------------------------
def test_all_channels_single_channel_no_channel_axis():
    """The historical crash: single-channel 2D + 'All channels'."""
    m = axes.to_canonical(np.zeros((100, 120)))
    arr, ch_axis = axes.as_segmentation_input(m, channel=None)
    assert arr.shape == (100, 120)  # the one plane, not misread as C
    assert ch_axis is None


def test_all_channels_multichannel_moves_axis_last():
    m = axes.to_canonical(np.zeros((3, 100, 120)))
    arr, ch_axis = axes.as_segmentation_input(m, channel=None)
    assert arr.shape == (100, 120, 3)
    assert ch_axis == -1


def test_all_channels_multichannel_zstack():
    m = axes.to_canonical(np.zeros((2, 8, 100, 120)))
    arr, ch_axis = axes.as_segmentation_input(m, channel=None)
    assert arr.shape == (8, 100, 120, 2)
    assert ch_axis == -1


def test_select_specific_channel_2d():
    data = np.stack([np.full((10, 10), c) for c in range(3)])  # (3,10,10) CYX
    m = axes.to_canonical(data)
    arr, ch_axis = axes.as_segmentation_input(m, channel=1)
    assert arr.shape == (10, 10) and ch_axis is None
    assert (arr == 1).all()


def test_channel_plane_zstack_returns_3d():
    m = axes.to_canonical(np.zeros((2, 8, 10, 10)))
    assert axes.channel_plane(m, 0).shape == (8, 10, 10)


def test_channel_out_of_range():
    m = axes.to_canonical(np.zeros((100, 120)))
    try:
        axes.channel_plane(m, 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_crop_to_bbox_keeps_c_and_z():
    m = axes.to_canonical(np.zeros((2, 8, 100, 120)))
    c = axes.crop_to_bbox(m, (10, 20, 60, 90))
    assert c.data.shape == (2, 8, 50, 70)


# --------------------------------------------------------------------------
# roi
# --------------------------------------------------------------------------
def test_crops_from_square():
    square = np.array([[10, 10], [10, 40], [40, 40], [40, 10]], dtype=float)
    crops = roi.crops_from_shapes([square], (100, 100))
    assert len(crops) == 1
    c = crops[0]
    assert c.bbox == (10, 10, 40, 40)
    assert c.polygon_mask.shape == (30, 30)
    assert c.polygon_mask.any()


def test_crops_skip_degenerate():
    line = np.array([[0, 0], [10, 10]], dtype=float)  # only 2 vertices
    assert roi.crops_from_shapes([line], (100, 100)) == []


def test_crops_clip_to_bounds():
    big = np.array([[-20, -20], [-20, 200], [200, 200], [200, -20]], dtype=float)
    crops = roi.crops_from_shapes([big], (100, 100))
    assert crops[0].bbox == (0, 0, 100, 100)


def test_apply_polygon_2d_and_3d():
    labels2d = np.ones((4, 4), dtype=int)
    mask = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool
    )
    out = roi.apply_polygon(labels2d, mask)
    assert out.sum() == 4  # only the True region kept

    labels3d = np.ones((3, 4, 4), dtype=int)
    out3 = roi.apply_polygon(labels3d, mask)
    assert out3.sum() == 12  # 4 per z * 3 z


def test_paste_offsets_and_no_collision():
    composite = np.zeros((100, 100), dtype=np.int32)
    crop_a = np.zeros((30, 30), dtype=np.int32)
    crop_a[5:10, 5:10] = 1
    crop_a[15:20, 15:20] = 2
    off = roi.paste_into(composite, crop_a, (0, 0, 30, 30), id_offset=0)
    assert off == 2
    crop_b = np.zeros((30, 30), dtype=np.int32)
    crop_b[0:5, 0:5] = 1
    off = roi.paste_into(composite, crop_b, (50, 50, 80, 80), id_offset=off)
    # Second ROI's cell became id 3, not colliding with ROI-A ids 1,2.
    assert off == 3
    assert set(np.unique(composite)) == {0, 1, 2, 3}


# --------------------------------------------------------------------------
# projection / double-positive
# --------------------------------------------------------------------------
def test_min_projection_keeps_only_shared_signal():
    a = np.zeros((1, 1, 20, 20), dtype=np.float32)
    b = np.zeros((1, 1, 20, 20), dtype=np.float32)
    # channel data will be stacked as (C,Z,Y,X); build 2 channels
    ch = np.zeros((2, 20, 20), dtype=np.float32)
    ch[0, 2:8, 2:8] = 100      # A-only blob
    ch[1, 12:18, 12:18] = 100  # B-only blob
    ch[0, 5:15, 5:15] = 100    # shared
    ch[1, 5:15, 5:15] = 100    # shared
    m = axes.to_canonical(ch)  # (2,1,20,20)
    proj = projection.min_projection(m, 0, 1, normalize=False)
    assert proj.shape == (20, 20)
    # A-only and B-only regions should be ~0 (min with the dark channel).
    assert proj[3, 3] == 0
    assert proj[16, 16] == 0
    assert proj[8, 8] > 0  # shared region survives


def test_colocalize_overlap():
    labels_a = np.zeros((20, 20), dtype=int)
    labels_a[2:8, 2:8] = 1   # overlaps B
    labels_a[12:18, 2:8] = 2  # no B
    labels_b = np.zeros((20, 20), dtype=int)
    labels_b[2:8, 2:8] = 5
    out = projection.colocalize(labels_a, labels_b, min_overlap=0.5)
    assert set(np.unique(out)) == {0, 1}  # cell 2 dropped


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------
def test_count_labels_distinct_nonzero():
    labels = np.array([0, 0, 1, 1, 5, 5, 5, 9])
    assert counting.count_labels(labels) == 3


def test_run_result_total():
    rr = counting.RunResult(
        labels=np.zeros((4, 4)),
        per_roi=[
            counting.RoiResult(0, 3, [1, 2, 3], (0, 0, 2, 2)),
            counting.RoiResult(1, 2, [4, 5], (2, 2, 4, 4)),
        ],
        backend="local",
        mode="single",
        run_meta={},
    )
    assert rr.total == 5


# --------------------------------------------------------------------------
# remote size filtering (the giant-blob fix)
# --------------------------------------------------------------------------
def test_filter_labels_drops_oversized_blob():
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[0:70, 0:70] = 1        # 4900 px = 49% of image -> a "blob"
    labels[80:85, 80:85] = 2      # 25 px -> a normal cell
    labels[90:96, 90:96] = 3      # 36 px -> a normal cell
    out = filter_labels(labels, min_size=15, max_size_fraction=0.4)
    assert set(np.unique(out)) == {0, 2, 3}  # blob (49% > 40%) removed


def test_filter_labels_drops_tiny():
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[0:10, 0:10] = 1  # 100 px
    labels[20:22, 20:21] = 2  # 2 px -> junk
    out = filter_labels(labels, min_size=15, max_size_fraction=0.4)
    assert set(np.unique(out)) == {0, 1}


def test_filter_labels_noop_when_all_in_range():
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[0:5, 0:5] = 1
    labels[10:16, 10:16] = 2
    out = filter_labels(labels, min_size=1, max_size_fraction=1.0)
    assert set(np.unique(out)) == {0, 1, 2}


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
