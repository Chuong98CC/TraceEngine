"""CPU-only unit tests for the Step-3a post-detection filters
(tools/general_test/pipeline/run_object_detection.py).

RexOmni struggles with the left/right robot-arm prompts of the dataset:
a side prompt (e.g. "left robot arm's black grippers") sometimes yields one
box per arm, and one hand occasionally yields two nearby boxes. The helpers
under test hard-filter those per-category detection lists:

- ``_merge_duplicate_boxes`` — boxes of one category whose centers share an
  image half and lie within ``DUP_MERGE_FRACTION`` of the image width are
  duplicates of the same instance and collapse into their union box
  (repeated until no qualifying pair remains).
- ``_category_side`` / ``_keep_side_box`` — a category whose prompt names
  exactly one side keeps only the extreme box on that side.
- ``_refine_detections`` — per-category orchestration (merge first, then
  side filter) plus human-readable notes of what fired.

Boxes mirror the wrapper's schema: {"type": "box", "coords": [x0, y0,
x1, y1]} in absolute pixels of the input image (640-wide here).
"""

from __future__ import annotations

import pytest

from tools.general_test.pipeline.run_object_detection import (
    DUP_MERGE_FRACTION,
    _category_side,
    _keep_side_box,
    _merge_duplicate_boxes,
    _refine_detections,
)

LEFT_GREY = "left robot arm's black grippers"

#: real same-hand duplicate pair (ep000000, subtask 1, frame 367): two
#: overlapping left-half boxes of the same left gripper.
DUP_PAIR = [
    {"type": "box", "coords": [0.0, 253.2, 71.75, 396.9]},
    {"type": "box", "coords": [51.25, 279.2, 187.07, 416.6]},
]
#: real left-arm + spurious right-arm pair (ep000000, subtask 0, frame 0):
#: one box per image half.
LEFT_AND_RIGHT = [
    {"type": "box", "coords": [0.0, 221.5, 80.72, 341.14]},
    {"type": "box", "coords": [515.72, 206.61, 618.86, 325.77]},
]
WIDTH = 640


def box(x0, y0, x1, y1):
    return {"type": "box", "coords": [x0, y0, x1, y1]}


def cx(b):
    x0, _, x1, _ = b["coords"]
    return (x0 + x1) / 2


def assert_boxes(result, expected_coords):
    """One box per expected coord list, in order, schema preserved."""
    assert [b["coords"] for b in result] == expected_coords
    assert all(b["type"] == "box" for b in result)


# --- _merge_duplicate_boxes -------------------------------------------------


def test_merge_same_hand_pair_into_union():
    """Two close boxes of the same left half are one hand: union box."""
    assert DUP_MERGE_FRACTION * WIDTH >= 86  # the pair's center distance
    merged = _merge_duplicate_boxes(DUP_PAIR, WIDTH)
    assert_boxes(merged, [[0.0, 253.2, 187.07, 416.6]])


def test_no_merge_across_image_halves():
    """A left-arm box and a right-arm box must not merge (side filter's job)."""
    merged = _merge_duplicate_boxes(LEFT_AND_RIGHT, WIDTH)
    assert_boxes(merged, [b["coords"] for b in LEFT_AND_RIGHT])


def test_no_merge_when_same_half_but_far_apart():
    """Distinct instances on the same side stay apart beyond the radius."""
    a = box(0, 0, 120, 120)     # center x = 60
    b = box(240, 0, 360, 120)   # center x = 300 (240 px away)
    merged = _merge_duplicate_boxes([a, b], WIDTH)
    assert len(merged) == 2


def test_merge_collapses_a_three_box_cluster():
    """Repeated merging collapses a tight 3-box cluster into one union."""
    cluster = [box(10, 0, 70, 100), box(90, 0, 150, 100), box(170, 0, 230, 100)]
    merged = _merge_duplicate_boxes(cluster, WIDTH)
    assert_boxes(merged, [[10, 0, 230, 100]])


def test_merge_leaves_single_boxes_untouched():
    merged = _merge_duplicate_boxes([box(0, 0, 100, 100)], WIDTH)
    assert_boxes(merged, [[0, 0, 100, 100]])


def test_merge_radius_scales_with_image_width():
    """The 0.2 * width radius is relative: a wider frame needs a wider gap
    before two boxes stop being duplicates."""
    pair = [box(100, 0, 160, 60), box(165, 0, 225, 60)]  # centers 130 / 195
    assert len(_merge_duplicate_boxes(pair, 640)) == 1   # 65 <= 0.2 * 640
    assert len(_merge_duplicate_boxes(pair, 300)) == 2   # 65 > 0.2 * 300


# --- _category_side ----------------------------------------------------------


@pytest.mark.parametrize("category,expected", [
    (LEFT_GREY, -1),
    ("right robot arm's black grippers", +1),
    ("  RIGHT  hand  ", +1),          # case / whitespace insensitive
    ("brown cup", None),              # no side in the prompt
    ("both left and right grippers", None),  # ambiguous: no side to enforce
    ("robot arm", None),
])
def test_category_side(category, expected):
    assert _category_side(category) == expected


# --- _keep_side_box ----------------------------------------------------------


def test_keep_leftmost_box_for_left_prompt():
    left_hand = box(0, 100, 90, 400)     # center x = 45
    right_hand = box(520, 90, 630, 400)  # center x = 575
    kept = _keep_side_box([right_hand, left_hand], -1)
    assert_boxes(kept, [[0, 100, 90, 400]])


def test_keep_rightmost_box_for_right_prompt():
    left_hand = box(0, 100, 90, 400)
    right_hand = box(520, 90, 630, 400)
    kept = _keep_side_box([left_hand, right_hand], +1)
    assert_boxes(kept, [[520, 90, 630, 400]])


def test_keep_side_noop_on_a_single_box():
    """A lone wrong-side box (one arm only) is kept as-is."""
    right_hand = box(520, 90, 630, 400)
    kept = _keep_side_box([right_hand], -1)
    assert_boxes(kept, [[520, 90, 630, 400]])


# --- _refine_detections ------------------------------------------------------


def test_refine_merges_duplicates_of_a_side_prompt():
    """Same-hand pair under a left prompt: merged to one box, no side pick."""
    preds = {LEFT_GREY: DUP_PAIR}
    refined, notes = _refine_detections(preds, WIDTH)
    assert len(refined[LEFT_GREY]) == 1
    assert_boxes(refined[LEFT_GREY], [[0.0, 253.2, 187.07, 416.6]])
    assert notes[LEFT_GREY] == "merge 2->1"


def test_refine_keeps_only_the_prompted_side():
    """Left prompt, one box per arm: the right arm is filtered out."""
    preds = {LEFT_GREY: LEFT_AND_RIGHT}
    refined, notes = _refine_detections(preds, WIDTH)
    assert_boxes(refined[LEFT_GREY], [LEFT_AND_RIGHT[0]["coords"]])
    assert notes[LEFT_GREY] == "left-keep 2->1"


def test_refine_passes_side_less_categories_through():
    """An object category with one far-right box and no side prompt is
    untouched (its single box passes merge; no side filter applies)."""
    cup = [box(200, 0, 300, 100)]
    preds = {"brown cup": cup}
    refined, notes = _refine_detections(preds, WIDTH)
    assert refined == preds
    assert notes == {}


def test_refine_reports_both_filters_when_both_fire():
    """Two close left-hand boxes AND one right-arm box under a left prompt:
    duplicates merge first, then the left pick keeps the union box."""
    dup_pair = [box(0, 100, 90, 300), box(80, 110, 180, 320)]
    right_arm = box(520, 90, 630, 400)
    refined, notes = _refine_detections({LEFT_GREY: dup_pair + [right_arm]},
                                        WIDTH)
    assert_boxes(refined[LEFT_GREY], [[0, 100, 180, 320]])
    assert notes[LEFT_GREY] == "merge 3->2, left-keep 2->1"
