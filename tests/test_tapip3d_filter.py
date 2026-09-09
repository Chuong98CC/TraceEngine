"""The Step-4 trace filters in flow_models/tapip3d:
filter_visible_tracks keeps the tracked columns that are always visible,
or whose invisible stretches are short (< max_invisible_stems) and whose
reappearance — when the run is bounded by visible stems on both sides —
lands within max_reappear_ratio times the column's expected hidden
travel (its fastest per-stem pixel step while continuously visible,
times the gap's stems) of the last visible position, measured in the
pixels of the reappearance stem (both world positions project through
that stem's own pose, so camera motion between the stems cancels). A
column with no such travel evidence has a zero envelope, so any
reappearance gap fails it. Edge runs (invisible from the first stem
and/or through the last stem) have no reappearance side to measure, so
only the length criterion applies. Distances are pixel-space: the old
world-space criterion (metres) read the genuine travel of
briefly-occluded moving points as a track failure, dropping the movers
this pipeline wants to keep.
filter_static_pixel_tracks keeps only the columns that move in the
tracked camera's pixels: the max displacement of the column's
reprojection from the first visible stem to any later visible stem must
exceed min_motion_pixels. Each stem projects through that stem's own
pose, so depth-noise wander along the viewing ray or a whole-scene
Step-2 coordinate jump barely moves the reprojection of a static
point."""
import torch

from flow_models.tapip3d.utils import (
    Tapip3DStreamPT2,
    filter_visible_tracks,
    filter_static_pixel_tracks,
)


def _coords_motion(t, q, step=0.01):
    """(T, Q, 3) coords moving `step` m per row along x — the reappearance
    jump between rows a-1 and b+1 is (b - a + 2) * step."""
    rows = torch.arange(t, dtype=torch.float32).unsqueeze(1).expand(t, q)
    cols = torch.arange(q, dtype=torch.float32).unsqueeze(0).expand(t, q)
    z = torch.zeros(t, q)
    return torch.stack([rows * step, cols, z], dim=-1)


def _cameras(t):
    """Per-stem identity extrinsics (w2c == world) and f=100, no-offset
    intrinsics: px = 100 * x / z, py = 100 * y / z (t stems)."""
    intr = torch.zeros(t, 3, 3)
    intr[:, 0, 0] = 100.0
    intr[:, 1, 1] = 100.0
    intr[:, 2, 2] = 1.0
    extr = torch.eye(4).expand(t, 4, 4).clone()
    return intr, extr


def _camera_visible(coords, visibs, **kw):
    """filter_visible_tracks with identity cameras (see _cameras)."""
    t = visibs.shape[0]
    intr, extr = _cameras(t)
    return filter_visible_tracks(coords, visibs, intr, extr, **kw)


def _visible_row(vis: list[bool], t=None):
    t = len(vis) if t is None else t
    return torch.tensor(vis, dtype=torch.bool).unsqueeze(1)


# ---------------------------------------------------------------------------
# filter_visible_tracks (pixel-space reappearance)
# ---------------------------------------------------------------------------

def test_all_visible_kept():
    coords = _coords_motion(6, 2)
    coords[..., 2] = 1.0                    # projectable depth
    visibs = torch.ones(6, 2, dtype=torch.bool)
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [True, True]
    assert reasons == [None, None]


def test_short_bounded_blip_kept():
    # 2 invisible stems between visible ones: the column's fastest visible
    # step is 0.1px/stem, so its 3-stem hidden travel envelope is 3x
    # 0.3px = 0.9px and the 0.3px reappearance gap stays under it
    coords = _coords_motion(6, 1, step=0.001)
    coords[:, 0, 2] = 1.0
    visibs = _visible_row([True, True, False, False, True, True])
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_long_run_dropped():
    # 8 invisible stems: the threshold is strict — a run must be < 8
    coords = _coords_motion(12, 1)
    coords[:, 0, 2] = 1.0
    visibs = _visible_row([True, False, False, False, False, False, False,
                           False, False, True, True, True])
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [False]
    assert "invisible" in reasons[0]


def test_fast_mover_blip_survives_self_relative():
    # the real-data case the self-relative rule exists for: a mover
    # sweeping ~100px/stem that blinks out for 1 stem. Its reappearance
    # gap (200px) is ~2x its own fastest visible step, far below the 3x
    # ratio -> kept, not read as a track snap
    coords = torch.zeros(8, 1, 3)
    coords[:, 0, 2] = 1.0
    coords[:, 0, 0] = torch.arange(8, dtype=torch.float32) * 1.0
    visibs = _visible_row([True, True, True, True, False, True, True, True])
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_slow_column_teleport_dropped():
    # a static column (fastest visible step ~0, so a ~0px envelope) that
    # snaps 100px away across 1 invisible stem: 100px >= 3x ~0
    coords = torch.zeros(6, 1, 3)
    coords[:, 0, 2] = 1.0
    coords[3:, 0, 0] = 1.0                  # 100px snap at z=1, f=100
    visibs = _visible_row([True, True, False, True, True, True])
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [False]
    assert "reappears" in reasons[0]


def test_evidence_free_reappearance_dropped():
    # 1 invisible stem, reappears 5px away with no visible travel evidence
    # anywhere (fastest step 0): the zero envelope leaves no allowance —
    # there is no absolute pixel floor anymore, so the reappearance reads
    # as a snap and the column drops
    coords = torch.zeros(4, 1, 3)
    coords[:, 0, 2] = 1.0
    coords[2:, 0, 0] = 0.05                 # 5px
    visibs = _visible_row([True, False, True, True])
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [False]
    assert reasons[0].startswith("reappears 5.0 px away")


def test_moving_column_with_blip_kept():
    # a mover travelling 3px/stem that blinks out for 1 stem: its
    # reappearance gap (6px) is genuine motion, not a track failure
    coords = torch.zeros(7, 1, 3)
    coords[:, 0, 2] = 1.0
    coords[:, 0, 0] = torch.arange(7, dtype=torch.float32) * 0.03
    visibs = _visible_row([True, True, True, False, True, True, True])
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_custom_reappear_ratio():
    # a column drifting 1.25px/stem (fastest visible step 1.25px) that
    # blinks out for 1 stem and reappears 5px off its motion trend: the
    # allowance is the ratio x 1.25px/stem x 2 gap stems — ratio 1.0
    # (2.5px) drops it (5 >= 2.5, strict) and ratio 5.0 (12.5px) keeps
    # it (5 < 12.5)
    coords = torch.zeros(5, 1, 3)
    coords[:, 0, 2] = 1.0
    coords[:, 0, 0] = torch.tensor([0.0, 0.0125, 0.025, 0.025, 0.075])
    visibs = _visible_row([True, True, True, False, True])
    keep, reasons = _camera_visible(coords, visibs, max_reappear_ratio=1.0)
    assert keep.tolist() == [False]
    assert reasons[0].startswith("reappears 5.0 px away")
    keep, _ = _camera_visible(coords, visibs, max_reappear_ratio=5.0)
    assert keep.tolist() == [True]


def test_camera_motion_between_stems_cancels():
    # the point travels 2px/stem in y while staying world-static in x at
    # 0.3m; the camera jumps 0.3m sideways by the reappearance stem
    # (b+1). Both positions project through the b+1 pose, so the 30px
    # parallax of the static x cancels and the measured gap is the 4px
    # of genuine y travel — under the 3x envelope of 2px/stem x 2 gap
    # stems (12px) the column is kept; measured through the a-1 pose the
    # same gap would read ~30px and drop the column
    coords = torch.zeros(6, 1, 3)
    coords[:, 0, 2] = 1.0
    coords[:, 0, 0] = 0.30                   # world-static x (30px parallax)
    coords[:, 0, 1] = torch.arange(6, dtype=torch.float32) * 0.02
    visibs = _visible_row([True, True, True, False, True, True])
    intr = torch.zeros(6, 3, 3)
    intr[:, 0, 0] = intr[:, 1, 1] = intr[:, 2, 2] = 100.0
    extr = torch.eye(4).expand(6, 4, 4).clone()
    extr[4, 0, 3] = 0.30                     # the b+1 camera at world x=0.30
    keep, reasons = filter_visible_tracks(coords, visibs, intr, extr)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_short_trailing_run_kept():
    # invisible through the last stem: length-only check (2 < 8)
    coords = _coords_motion(5, 1)
    coords[:, 0, 2] = 1.0
    visibs = _visible_row([True, True, True, False, False])
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_long_trailing_run_dropped():
    coords = _coords_motion(10, 1)
    coords[:, 0, 2] = 1.0
    visibs = _visible_row([True, True, False, False, False, False, False,
                           False, False, False])
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [False]
    assert "invisible" in reasons[0]


def test_short_leading_run_kept():
    # invisible from the first stem, then visible: no previous visible
    # stem to measure displacement from -> length-only check
    coords = _coords_motion(5, 1)
    coords[:, 0, 2] = 1.0
    visibs = _visible_row([False, False, True, True, True])
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_all_invisible_short_trace_kept():
    # one run spanning both edges, shorter than the threshold
    coords = _coords_motion(3, 1)
    coords[:, 0, 2] = 1.0
    visibs = torch.zeros(3, 1, dtype=torch.bool)
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_one_bad_run_drops_the_column():
    # a fine 1-stem blip (reappears close) followed by an 8-stem run
    # -> the column is dropped by the long run
    coords = _coords_motion(13, 1, step=0.001)
    coords[:, 0, 2] = 1.0
    visibs = _visible_row([True, False, True, True, False, False, False,
                           False, False, False, False, False, True])
    keep, _ = _camera_visible(coords, visibs)
    assert keep.tolist() == [False]


def test_raw_logits_rejected():
    # un-thresholded logits must not silently cast to bool (any nonzero
    # logit would look visible and the filter would drop nothing)
    coords = _coords_motion(3, 1)
    logits = torch.tensor([-2.0, 2.0, 2.0])
    intr, extr = _cameras(3)
    try:
        filter_visible_tracks(coords, logits, intr, extr)
    except TypeError:
        return
    raise AssertionError("expected TypeError for float visibs")


def test_mixed_columns_and_reasons():
    coords = _coords_motion(9, 3)
    coords[:, 0, 2] = 1.0
    coords[:, 1, 2] = 1.0
    coords[:, 2, 2] = 1.0
    visibs = torch.tensor(
        [[True, False, True],
         [True, False, True],
         [True, False, True],
         [True, False, True],
         [True, False, True],
         [True, False, True],
         [True, False, True],
         [True, False, True],
         [True, False, True]], dtype=torch.bool)
    keep, reasons = _camera_visible(coords, visibs)
    assert keep.tolist() == [True, False, True]
    assert reasons == [None, "invisible 9 stems (>= 8)", None]


def test_custom_run_length_threshold():
    # max_invisible_stems=1: a single invisible stem is already a drop
    coords = _coords_motion(4, 2)
    coords[:, :, 2] = 1.0
    visibs = torch.tensor([[True, False],
                           [True, True],
                           [True, True],
                           [True, True]], dtype=torch.bool)
    keep, reasons = _camera_visible(coords, visibs, max_invisible_stems=1)
    assert keep.tolist() == [True, False]
    assert reasons[1] == "invisible 1 stems (>= 1)"


def test_visible_camera_shapes_validated():
    coords = torch.zeros(3, 1, 3)
    coords[:, 0, 2] = 1.0
    visibs = torch.ones(3, 1, dtype=torch.bool)
    intr, extr = _cameras(3)
    for bad_intr, bad_extr in [
            (intr[:2], extr),              # wrong stem count
            (torch.zeros(3, 2, 3), extr),  # wrong intrinsics shape
            (intr, extr[:, :3]),           # wrong extrinsics shape
    ]:
        try:
            filter_visible_tracks(coords, visibs, bad_intr, bad_extr)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for a bad camera shape")


# ---------------------------------------------------------------------------
# filter_static_pixel_tracks
# ---------------------------------------------------------------------------

def test_pixel_static_point_dropped():
    coords = torch.zeros(5, 1, 3)          # world-static at z=1
    coords[:, 0, 2] = 1.0
    visibs = torch.ones(5, 1, dtype=torch.bool)
    intr, extr = _cameras(5)
    keep, reasons = filter_static_pixel_tracks(coords, visibs, intr, extr)
    assert keep.tolist() == [False]
    assert reasons == ["static (pixel): max displacement 0.0 px (<= 3.0 px)"]


def test_pixel_transverse_mover_kept():
    # world x moves 5cm at z=1: 100 * 0.05 = 5px > 3px
    coords = torch.zeros(5, 1, 3)
    coords[:, 0, 2] = 1.0
    coords[2:, 0, 0] = 0.05
    visibs = torch.ones(5, 1, dtype=torch.bool)
    intr, extr = _cameras(5)
    keep, reasons = filter_static_pixel_tracks(coords, visibs, intr, extr)
    assert keep.tolist() == [True]
    assert reasons == [None]


def test_pixel_ray_noise_mover_dropped():
    # 5cm of world motion ALONG the viewing ray (depth noise): at 10px off
    # center the reprojected pixel moves ~0.5px -> static in pixels, even
    # though the metric (world) displacement is 5cm
    coords = torch.zeros(5, 1, 3)
    coords[:, 0, 0] = 0.10                 # 10px off center at z=1
    coords[:, 0, 2] = torch.linspace(1.0, 1.05, 5)   # z wanders 5cm
    visibs = torch.ones(5, 1, dtype=torch.bool)
    intr, extr = _cameras(5)
    keep, reasons = filter_static_pixel_tracks(coords, visibs, intr, extr)
    assert keep.tolist() == [False]
    assert "static (pixel)" in reasons[0]


def test_pixel_displacement_at_exactly_threshold_dropped():
    # strict criterion: exactly min_motion_pixels does not keep (float64)
    coords = torch.zeros(2, 1, 3, dtype=torch.float64)
    coords[:, 0, 2] = 1.0
    coords[1, 0, 0] = 0.03                 # exactly 3.0 px at f=100
    visibs = torch.ones(2, 1, dtype=torch.bool)
    intr, extr = _cameras(2)
    keep, _ = filter_static_pixel_tracks(coords, visibs, intr, extr,
                                         min_motion_pixels=3.0)
    assert keep.tolist() == [False]


def test_pixel_never_visible_column_dropped():
    coords = torch.zeros(5, 1, 3)
    coords[:, 0, 2] = 1.0
    visibs = torch.zeros(5, 1, dtype=torch.bool)
    intr, extr = _cameras(5)
    keep, reasons = filter_static_pixel_tracks(coords, visibs, intr, extr)
    assert keep.tolist() == [False]
    assert reasons == ["no visible stem (cannot measure motion)"]


def test_pixel_custom_threshold():
    # 5px mover kept at 2px threshold, dropped at 8px
    coords = torch.zeros(5, 1, 3)
    coords[:, 0, 2] = 1.0
    coords[1:, 0, 0] = 0.05
    visibs = torch.ones(5, 1, dtype=torch.bool)
    intr, extr = _cameras(5)
    keep, _ = filter_static_pixel_tracks(coords, visibs, intr, extr,
                                         min_motion_pixels=2.0)
    assert keep.tolist() == [True]
    keep, reasons = filter_static_pixel_tracks(coords, visibs, intr, extr,
                                               min_motion_pixels=8.0)
    assert keep.tolist() == [False]
    assert reasons == ["static (pixel): max displacement 5.0 px (<= 8.0 px)"]


def test_pixel_camera_shapes_validated():
    coords = torch.zeros(3, 1, 3)
    visibs = torch.ones(3, 1, dtype=torch.bool)
    intr, extr = _cameras(3)
    for bad_intr, bad_extr in [
            (intr[:2], extr),              # wrong stem count
            (torch.zeros(3, 2, 3), extr),  # wrong intrinsics shape
            (intr, extr[:, :3]),           # wrong extrinsics shape
    ]:
        try:
            filter_static_pixel_tracks(coords, visibs, bad_intr, bad_extr)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for a bad camera shape")


def test_run_filter_static_pixel_ok():
    # the pixel filter alone runs without any scene reference: only the
    # reprojected pixels under each stem's own cameras are measured
    si = Tapip3DStreamPT2(_StubStreamPT2(), torch.zeros(4, 4), device="cpu")
    out = si.run(iter(()), 0, filter_static_pixel=True)
    coords, visibs, keep, reasons = out
    assert keep.tolist() == [True, True, True, True]
    assert reasons == [None] * 4
    assert coords.shape == (0, 4, 3) and visibs.shape == (0, 4)


# ---------------------------------------------------------------------------
# Tapip3DStreamPT2.run: filter plumbing
# ---------------------------------------------------------------------------

class _StubStreamPT2:
    """The graph attributes Tapip3DStreamPT2.__init__ reads — enough to
    drive run() on CPU with an empty batch stream."""
    num_queries = 4
    seq_len = 4


def test_run_filter_visible_alone_collects_geometry():
    # filter_visible now measures reappearances in pixels, so run() must
    # collect the per-stem cameras even when no pixel static filter is on
    si = Tapip3DStreamPT2(_StubStreamPT2(), torch.zeros(4, 4), device="cpu")
    out = si.run(iter(()), 0, filter_visible=True)
    coords, visibs, keep, reasons = out
    assert keep.tolist() == [True, True, True, True]
    assert reasons == [None] * 4
    assert coords.shape == (0, 4, 3) and visibs.shape == (0, 4)
