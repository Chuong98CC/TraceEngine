"""CPU-only unit tests for Step 3b's near-object manipulator sampling
(tools/general_test/pipeline/run_object_init_points.py).

--sampling-mode no_roma samples the manipulator's top-k points inside its
row-0 mask (the SAM ∪ optical-flow union under --with-optical-flow-mask).
The points are no longer drawn uniformly: every mask pixel is weighted by
its distance to the manipulated object's Step-3a bbox center —

    R = median(d)                  (d in pixels: the cut radius)
    w = 1 / (1 + (d/R)**2)         (d measured in cut radii)
    w = w / w.max()                (max weight 1)
    w = 0 where w < median(w)      (exactly the pixels beyond R)
    p = w / w.sum()                (drawn without replacement)

so the draw stays on the half of the arm mask nearest the object, spread
over it and at most 2x denser toward it. The pure pieces are under test:

- ``_object_center`` — the object prompt's largest-area Step-3a box on the
  manipulator's sampled key-frame (the first entry of the candidate list),
  falling back to the first key-frame that has one.
- ``_proximity_weights`` — the 1/(1+(d/R)**2) weights (R the pool's median
  distance), max-normalized to 1.
- ``_near_object_probs`` — the below-median cut + normalization to a
  probability vector aligned with the mask's nonzero pixels.
- ``_sample_in_mask`` — the draw itself; with ``p=None`` it is the uniform
  baseline, point-for-point identical to the previous implementation.

Boxes mirror the Step-3a schema ({"type": "box", "coords": [x0, y0, x1,
y1]}, absolute pixels) — see tests/test_object_detection_filter.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.general_test.pipeline.run_object_init_points import (
    _near_object_probs,
    _object_center,
    _proximity_weights,
    _sample_in_mask,
)

OBJECT = "brown cup"
MANIPULATOR = "robot arm's black grippers"
#: the object's two boxes on the sampled frame: the smaller one is a stray
#: detection, the larger one (area 6400 vs 1600) is what the center comes
#: from — the same "largest box of the prompt" pick the crop boxes use.
SMALL_BOX = [10.0, 20.0, 50.0, 60.0]
LARGE_BOX = [100.0, 100.0, 180.0, 180.0]
LARGE_CENTER = (140.0, 140.0)


# --- _object_center ---------------------------------------------------------

def test_object_center_takes_the_largest_box_of_the_sampled_frame():
    seg_dets = {str(338): {OBJECT: [
        {"type": "box", "coords": SMALL_BOX},
        {"type": "box", "coords": LARGE_BOX},
        # non-box predictions of the same prompt are ignored
        {"type": "score", "value": 0.9},
    ]}}
    assert _object_center(seg_dets, [338, 169, 0], OBJECT) == \
        (*LARGE_CENTER, 338)


def test_object_center_prefers_the_sampled_frame_over_later_ones():
    """The manipulator samples the first key-frame, so its box wins even
    when a later key-frame carries a bigger one."""
    seg_dets = {str(338): {OBJECT: [{"type": "box", "coords": SMALL_BOX}]},
                str(169): {OBJECT: [{"type": "box", "coords": LARGE_BOX}]}}
    assert _object_center(seg_dets, [338, 169], OBJECT) == \
        (30.0, 40.0, 338)


def test_object_center_falls_back_to_the_first_frame_that_has_a_box():
    seg_dets = {str(338): {OBJECT: []},
                str(169): {OBJECT: [{"type": "box", "coords": LARGE_BOX}]},
                str(0): {OBJECT: [{"type": "box", "coords": SMALL_BOX}]}}
    assert _object_center(seg_dets, [338, 169, 0], OBJECT) == \
        (*LARGE_CENTER, 169)


def test_object_center_is_none_without_a_reference():
    seg_dets = {str(338): {OBJECT: [], MANIPULATOR: [
        {"type": "box", "coords": LARGE_BOX}]}}
    assert _object_center(seg_dets, [338, 169], OBJECT) is None      # no boxes
    assert _object_center(seg_dets, [338, 169], MANIPULATOR) == \
        (*LARGE_CENTER, 338)                                    # other prompt
    assert _object_center(seg_dets, [338], None) is None       # no object prompt
    assert _object_center(seg_dets, [], OBJECT) is None            # no key-frames
    assert _object_center(None, [338], OBJECT) is None           # no detections


# --- _proximity_weights -----------------------------------------------------

def test_proximity_weights_are_normalized_and_decay_with_distance():
    xs = np.array([0.0, 3.0, 4.0, 30.0])
    ys = np.array([0.0, 4.0, 3.0, 40.0])
    w = _proximity_weights(xs, ys, (0.0, 0.0))
    # d = 0, 5, 5, 50 -> R = median(d) = 5 -> 1/(1+(d/5)^2), max-normalized
    assert w[0] == pytest.approx(1.0)                # the nearest pixel: max
    assert w.max() == pytest.approx(1.0)
    assert w[1] == pytest.approx(1.0 / 2.0)          # the cut radius: half
    assert w[2] == pytest.approx(1.0 / 2.0)
    assert w[3] == pytest.approx(1.0 / 101.0)
    assert (w > 0).all()                          # never zero: the pool holds
    assert w[1] > w[3]                               # monotone in d


def test_proximity_weights_max_out_at_the_pixel_nearest_the_center():
    """The normalization is per pool: whichever mask pixel is closest to
    the center carries weight 1, the rest scale down from there."""
    xs = np.array([100.0, 200.0, 50.0])
    ys = np.array([100.0, 100.0, 100.0])
    w = _proximity_weights(xs, ys, (100.0, 100.0))
    # d = 0, 100, 50 -> R = 50
    assert w[0] == pytest.approx(1.0)                          # d = 0
    assert w[2] == pytest.approx(1.0 / (1.0 + 1.0 ** 2))       # d = R: half
    assert w[1] == pytest.approx(1.0 / (1.0 + 2.0 ** 2))       # d = 2R


def test_proximity_weights_spread_over_the_kept_half():
    """Measuring d in cut radii is what stops the draw from collapsing
    onto the pixels nearest the center: within the kept half (d <= R) the
    weights span [0.5, 1], a density ratio of at most 2 — where raw pixels
    would make the nearest pixel ~100x heavier than the cut radius one."""
    ys, xs = np.mgrid[0:200, 0:200]
    xs, ys = xs.ravel().astype(float), ys.ravel().astype(float)
    d = np.hypot(xs - 100.0, ys - 100.0)
    keep = d <= np.median(d)
    w = _proximity_weights(xs, ys, (100.0, 100.0))
    assert w[keep].min() >= 0.49          # the cut radius still weighs ~half
    assert w[keep].max() / w[keep].min() <= 2.05


def test_proximity_weights_of_a_point_pool_are_ones():
    """A pool entirely on the center has no radius to scale by."""
    w = _proximity_weights(np.array([5.0, 5.0]), np.array([5.0, 5.0]),
                           (5.0, 5.0))
    assert w == pytest.approx(np.ones(2))


def test_proximity_weights_of_an_empty_pool_are_empty():
    assert _proximity_weights(np.zeros(0), np.zeros(0), (10.0, 10.0)).size == 0


# --- _near_object_probs -----------------------------------------------------

def test_near_object_probs_drop_the_farther_half():
    w = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.05])
    p = _near_object_probs(w)
    assert p.sum() == pytest.approx(1.0)
    # median 0.5: the three heavier pixels survive, proportionally to w
    assert (p[3:] == 0).all()
    assert p[:3] == pytest.approx(np.array([1.0, 0.8, 0.6]) / 2.4)


def test_near_object_probs_keep_exactly_the_at_least_median_pixels():
    w = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    p = _near_object_probs(w)
    assert np.count_nonzero(p) == np.count_nonzero(w >= np.median(w)) == 3


def test_near_object_probs_keep_exactly_the_pixels_within_the_cut_radius():
    """The median cut and the weight scale are the same quantity: scaling d
    by R leaves the kept set untouched (the weights' median sits at R), it
    only changes how the draw spreads inside it."""
    ys, xs = np.mgrid[0:200, 0:200]
    xs, ys = xs.ravel().astype(float), ys.ravel().astype(float)
    center = (60.0, 90.0)
    d = np.hypot(xs - center[0], ys - center[1])
    p = _near_object_probs(_proximity_weights(xs, ys, center))
    assert np.array_equal(p > 0, d <= np.median(d))


def test_near_object_probs_of_equal_weights_stay_uniform():
    p = _near_object_probs(np.full(4, 0.25))
    assert p == pytest.approx(np.full(4, 0.25))


def test_near_object_probs_of_a_single_pixel():
    assert _near_object_probs(np.array([1.0])) == pytest.approx(np.array([1.0]))


# --- _sample_in_mask --------------------------------------------------------

def test_sample_in_mask_without_weights_is_the_uniform_baseline():
    """p=None must keep the previous draw exactly — same seed, same points
    (the shipped no_roma outputs stay reproducible)."""
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:40, 20:60] = True
    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(7)
    expected = rng.choice(len(xs), size=32, replace=False)
    got = _sample_in_mask(mask, 32, 7)
    assert got == pytest.approx(
        np.stack([xs[expected], ys[expected]], axis=1).astype(np.float32))


def test_sample_in_mask_draws_distinct_points_inside_the_mask():
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:45, 5:45] = True
    pts = _sample_in_mask(mask, 40, 3)
    assert pts.shape == (40, 2)
    assert len({tuple(p) for p in pts}) == 40           # distinct pixels
    assert mask[pts[:, 1].astype(int), pts[:, 0].astype(int)].all()


def test_sample_in_mask_caps_at_the_mask_pixels():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :3] = True
    assert len(_sample_in_mask(mask, 64, 1)) == 3
    assert len(_sample_in_mask(np.zeros((10, 10), dtype=bool), 64, 1)) == 0


def test_sample_in_mask_is_deterministic():
    mask = np.zeros((40, 40), dtype=bool)
    mask[4:36, 4:36] = True
    ys, xs = np.nonzero(mask)
    p = _near_object_probs(_proximity_weights(xs, ys, (10.0, 10.0)))
    assert _sample_in_mask(mask, 16, 5, p) == \
        pytest.approx(_sample_in_mask(mask, 16, 5, p))


# --- weighted draw end to end (synthetic mask) ------------------------------

def _disc_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2


def test_weighted_draw_concentrates_near_the_object():
    """The behavioural claim: with the object center at one edge of a large
    mask, every drawn point lands in the mask's nearer half and the draw is
    pulled further in than that half's own median distance."""
    mask = _disc_mask(240, 240, 120.0, 120.0, 100.0)
    center = (20.0, 20.0)                     # object near the mask's corner
    ys, xs = np.nonzero(mask)
    d_mask = np.hypot(xs - center[0], ys - center[1])
    p = _near_object_probs(_proximity_weights(xs, ys, center))
    pts = _sample_in_mask(mask, 64, 11, p)
    d_pts = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
    assert len(pts) == 64
    assert d_pts.max() <= np.median(d_mask)          # the cut is a hard bound
    assert d_pts.mean() < np.median(d_mask)          # and the draw is weighted
    # ... clearly nearer than the uniform baseline over the same mask
    uniform = _sample_in_mask(mask, 64, 11)
    d_uniform = np.hypot(uniform[:, 0] - center[0], uniform[:, 1] - center[1])
    assert d_pts.mean() < 0.75 * d_uniform.mean()
