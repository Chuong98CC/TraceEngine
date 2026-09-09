"""The Step-4 trace filters in flow_models/tapip3d:
filter_visible_tracks keeps the tracked columns that are always visible,
or whose invisible stretches are short (< max_invisible_stems) and — when
the point reappears (the run is bounded by visible stems on both sides) —
reappears within max_reappear_displacement metres of where they were last
visible. Edge runs (invisible from the first stem and/or through the last
stem) have no reappearance side, so only the length criterion applies.
filter_static_tracks keeps only the columns that move: the max
displacement from the first visible (first-appear) stem to any later
visible stem must exceed min_motion_displacement metres."""
import torch

from flow_models.tapip3d.utils import (
    filter_visible_tracks, filter_static_tracks,
)


def _coords_motion(t, q, step=0.01):
    """(T, Q, 3) coords moving `step` m per row along x — the reappearance
    jump between rows a-1 and b+1 is (b - a + 2) * step."""
    rows = torch.arange(t, dtype=torch.float32).unsqueeze(1).expand(t, q)
    cols = torch.arange(q, dtype=torch.float32).unsqueeze(0).expand(t, q)
    z = torch.zeros(t, q)
    return torch.stack([rows * step, cols, z], dim=-1)


def test_all_visible_kept():
    coords = _coords_motion(6, 2)
    visibs = torch.ones(6, 2, dtype=torch.bool)
    keep, reasons = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [True, True]
    assert reasons == [None, None]


def test_short_bounded_blip_kept():
    # 2 invisible stems between visible ones, displacement (a-1 -> b+1)
    # = 3 rows * 1mm = 3mm < 1cm
    coords = _coords_motion(6, 1, step=0.001)
    visibs = torch.tensor([True, True, False, False, True, True],
                          dtype=torch.bool).unsqueeze(1)
    keep, reasons = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_long_run_dropped():
    # 4 invisible stems: the threshold is strict — a run must be < 4
    coords = _coords_motion(8, 1)
    visibs = torch.tensor([True, False, False, False, False, True, True,
                           True], dtype=torch.bool).unsqueeze(1)
    keep, reasons = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [False]
    assert "invisible" in reasons[0]


def test_reappearance_jump_dropped():
    # 1 invisible stem, but the point reappears 2cm away (> 1cm)
    coords = _coords_motion(5, 1, step=0.01)
    visibs = torch.tensor([True, False, True, True, True],
                          dtype=torch.bool).unsqueeze(1)
    keep, reasons = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [False]
    assert "reappears" in reasons[0]


def test_reappearance_at_exactly_threshold_dropped():
    # displacement == max_reappear_displacement is not less than it
    # (float64: an exact comparison at the boundary)
    coords = torch.tensor([[[0.0, 0.0, 0.0]], [[0.01, 0.0, 0.0]],
                           [[0.01, 0.0, 0.0]]], dtype=torch.float64)
    visibs = torch.tensor([True, False, True], dtype=torch.bool).unsqueeze(1)
    keep, _ = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [False]


def test_short_trailing_run_kept():
    # invisible through the last stem: length-only check (2 < 4)
    coords = _coords_motion(5, 1)
    visibs = torch.tensor([True, True, True, False, False],
                          dtype=torch.bool).unsqueeze(1)
    keep, reasons = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_long_trailing_run_dropped():
    coords = _coords_motion(6, 1)
    visibs = torch.tensor([True, True, False, False, False, False],
                          dtype=torch.bool).unsqueeze(1)
    keep, reasons = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [False]
    assert "invisible" in reasons[0]


def test_short_leading_run_kept():
    # invisible from the first stem, then visible: no previous visible
    # stem to measure displacement from -> length-only check
    coords = _coords_motion(5, 1)
    visibs = torch.tensor([False, False, True, True, True],
                          dtype=torch.bool).unsqueeze(1)
    keep, reasons = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_all_invisible_short_trace_kept():
    # one run spanning both edges, shorter than the threshold
    coords = _coords_motion(3, 1)
    visibs = torch.zeros(3, 1, dtype=torch.bool)
    keep, reasons = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_one_bad_run_drops_the_column():
    # a fine 1-stem blip (reappears 2mm away) followed by a 4-stem run
    # -> the column is dropped by the long run
    coords = _coords_motion(9, 1, step=0.001)
    visibs = torch.tensor([True, False, True, True, False, False, False,
                           False, True], dtype=torch.bool).unsqueeze(1)
    keep, _ = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [False]


def test_raw_logits_rejected():
    # un-thresholded logits must not silently cast to bool (any nonzero
    # logit would look visible and the filter would drop nothing)
    coords = _coords_motion(3, 1)
    logits = torch.tensor([-2.0, 2.0, 2.0])
    try:
        filter_visible_tracks(coords, logits)
    except TypeError:
        return
    raise AssertionError("expected TypeError for float visibs")


def test_mixed_columns_and_reasons():
    coords = _coords_motion(5, 3)
    visibs = torch.tensor(
        [[True, False, True],
         [True, False, True],
         [True, False, True],
         [True, False, True],
         [True, False, True]], dtype=torch.bool)
    keep, reasons = filter_visible_tracks(coords, visibs)
    assert keep.tolist() == [True, False, True]
    assert reasons == [None, "invisible 5 stems (>= 4)", None]


def test_custom_thresholds():
    # max_invisible_stems=1: a single invisible stem is already a drop
    coords = _coords_motion(4, 2)
    visibs = torch.tensor([[True, False],
                           [True, True],
                           [True, True],
                           [True, True]], dtype=torch.bool)
    keep, reasons = filter_visible_tracks(coords, visibs,
                                          max_invisible_stems=1)
    assert keep.tolist() == [True, False]
    assert reasons[1] == "invisible 1 stems (>= 1)"

    # 5mm reappearance allowance rejects a 1cm jump
    coords = _coords_motion(3, 1, step=0.005)
    visibs = torch.tensor([True, False, True], dtype=torch.bool).unsqueeze(1)
    keep, _ = filter_visible_tracks(coords, visibs,
                                    max_reappear_displacement=0.005)
    assert keep.tolist() == [False]


# ---------------------------------------------------------------------------
# filter_static_tracks
# ---------------------------------------------------------------------------

def test_static_point_dropped():
    coords = _coords_motion(5, 1, step=0.0)
    visibs = torch.ones(5, 1, dtype=torch.bool)
    keep, reasons = filter_static_tracks(coords, visibs)
    assert keep.tolist() == [False]
    assert reasons == ["static: max displacement 0.000 m (<= 0.01 m)"]


def test_moving_point_kept():
    coords = _coords_motion(5, 1, step=0.01)   # 4cm of travel
    visibs = torch.ones(5, 1, dtype=torch.bool)
    keep, reasons = filter_static_tracks(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_displacement_at_exactly_threshold_dropped():
    # the criterion is strict: a point must move MORE than the threshold
    # (float64: an exact comparison at the boundary)
    coords = torch.tensor([[[0.0, 0.0, 0.0]], [[0.01, 0.0, 0.0]]],
                          dtype=torch.float64)
    visibs = torch.ones(2, 1, dtype=torch.bool)
    keep, _ = filter_static_tracks(coords, visibs)
    assert keep.tolist() == [False]


def test_first_appear_is_first_visible_stem():
    # invisible at stem 0 (coords there hold the query init position),
    # visible and static 5cm away from stem 1 on: anchored at the first
    # VISIBLE stem the point never moves -> dropped; the init position
    # is not an appearance
    coords = _coords_motion(5, 1, step=0.0)
    coords[1:, 0, 0] = 0.05
    visibs = torch.tensor([False, True, True, True, True],
                          dtype=torch.bool).unsqueeze(1)
    keep, reasons = filter_static_tracks(coords, visibs)
    assert keep.tolist() == [False]
    assert "static" in reasons[0]


def test_invisible_gap_positions_ignored():
    # static at the origin with a 9cm junk extrapolation while occluded:
    # displacement is measured on visible stems only, so it stays 0
    coords = torch.zeros(9, 1, 3)
    coords[3:6, 0, 0] = 0.09
    visibs = torch.tensor([True, True, True, False, False, False,
                           True, True, True], dtype=torch.bool).unsqueeze(1)
    keep, reasons = filter_static_tracks(coords, visibs)
    assert keep.tolist() == [False]
    assert "static" in reasons[0]


def test_never_visible_column_dropped():
    # no visible stem -> no first appearance to measure motion from
    coords = _coords_motion(5, 1, step=0.01)
    visibs = torch.zeros(5, 1, dtype=torch.bool)
    keep, reasons = filter_static_tracks(coords, visibs)
    assert keep.tolist() == [False]
    assert reasons == ["no visible stem (cannot measure motion)"]


def test_single_visible_stem_dropped():
    # one observation carries no motion evidence
    coords = _coords_motion(5, 1, step=0.0)
    visibs = torch.zeros(5, 1, dtype=torch.bool)
    visibs[2] = True
    keep, _ = filter_static_tracks(coords, visibs)
    assert keep.tolist() == [False]


def test_custom_motion_threshold():
    # col0 travels 8mm, col1 travels 2mm; a 5mm threshold keeps col0 only
    # (step=1.0 basis, then scaled per column along x)
    coords = _coords_motion(5, 2, step=1.0)
    coords[..., 0] *= torch.tensor([0.002, 0.0005])
    visibs = torch.ones(5, 2, dtype=torch.bool)
    keep, reasons = filter_static_tracks(coords, visibs,
                                         min_motion_displacement=0.005)
    assert keep.tolist() == [True, False]
    assert reasons == [None, "static: max displacement 0.002 m (<= 0.005 m)"]


def test_mixed_static_moving_and_never_visible():
    coords = torch.zeros(4, 3, 3)
    coords[1:, 1, 0] = 0.03          # col1 travels 3cm from stem 1 on
    visibs = torch.tensor([[True, True, False],
                           [True, True, False],
                           [True, True, False],
                           [True, True, False]], dtype=torch.bool)
    keep, reasons = filter_static_tracks(coords, visibs)
    assert keep.tolist() == [False, True, False]
    assert reasons == ["static: max displacement 0.000 m (<= 0.01 m)",
                       None,
                       "no visible stem (cannot measure motion)"]


def test_raw_logits_rejected_for_static():
    coords = _coords_motion(3, 1)
    logits = torch.tensor([-2.0, 2.0, 2.0])
    try:
        filter_static_tracks(coords, logits)
    except TypeError:
        return
    raise AssertionError("expected TypeError for float visibs")
