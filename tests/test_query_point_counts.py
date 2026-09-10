"""CPU-only unit tests for the 128-manipulator-query feature.

The static filter of Step 4 (--filter-static-pixel) removes the
stationary keypoints, so the manipulator pass needs more seeds to keep
enough moving survivors. The count is capped in two
places that must move together:

- Step 3b (tools/general_test/pipeline/run_object_init_points.py)
  samples at most ``--object-top-k`` keypoints per prompt into
  init_points.npz; ``--manipulator-top-k`` raises the cap for the
  manipulator-role prompts only (object and role-less prompts keep
  ``--object-top-k``).
- Step 4 (tools/astribot/run_step4_traces.py) caps the role keypoints of
  each pass; the manipulator role gets 128 slots, object/unlabelled keep
  64. The shipped TAPIP3D iteration graph is fixed at 1088 queries total,
  so the support block auto-trims to 1088 - role keypoints (960 for a
  full 128-query manipulator pass). The trim is query-aware: the grid
  cells whose anchor-frame pixels lie nearest the role keypoints are
  dropped (one cell per keypoint, cyclically in rank order, until the
  drop budget is spent), keeping the surviving support spread away from
  the tracked points — see ``_drop_cells_near_px``. Without query pixels
  the even-spread linspace trim remains the fallback.

The Step-3 driver (tools/astribot/run_step3_init_points.py) forwards
``--object-top-k`` and ``--manipulator-top-k`` (default 128) to Step 3b.
"""

from __future__ import annotations

import numpy as np
import torch

from flow_models.tapip3d.utils._sampling import get_points_on_a_grid
from tools.astribot.run_step3_init_points import (
    DEFAULT_MANIPULATOR_TOP_K,
    DEFAULT_OBJECT_TOP_K,
    _build_3b_cmd,
    parse_args,
)
from tools.astribot.run_step4_traces import (
    MAX_KEPT_KEYPOINTS,
    ROLE_MAX_KEYPOINTS,
    SUPPORT_GRID_SIZE,
    _cap_keypoint_survivors,
    _drop_cells_near_px,
    _role_keypoint_cap,
    _support_queries,
)
from tools.general_test.pipeline.run_object_init_points import _prompt_top_k


# --- Step 3b: role-aware per-prompt top-k ------------------------------------


def test_prompt_top_k_object_role_uses_top_k():
    assert _prompt_top_k(64, 128, "object") == 64


def test_prompt_top_k_unlabelled_role_uses_top_k():
    assert _prompt_top_k(64, 128, None) == 64


def test_prompt_top_k_manipulator_role_uses_manipulator_top_k():
    assert _prompt_top_k(64, 128, "manipulator") == 128


def test_prompt_top_k_manipulator_falls_back_to_top_k_when_unset():
    assert _prompt_top_k(64, None, "manipulator") == 64


# --- Step-3 driver ------------------------------------------------------------


def _driver_args(*extra: str):
    """parse_args without a program name (argparse's argv excludes it)."""
    return parse_args(["--repo-id", "r", "--data-root", "/tmp/x", *extra])


def test_driver_manipulator_top_k_defaults_to_128():
    args = _driver_args()
    assert args.object_top_k == DEFAULT_OBJECT_TOP_K == 64
    assert args.manipulator_top_k == DEFAULT_MANIPULATOR_TOP_K == 128


def test_driver_3b_cmd_forwards_both_k_flags():
    args = _driver_args()
    cmd = _build_3b_cmd(args, repo_root=__import__("pathlib").Path("/repo"))
    assert cmd[cmd.index("--object-top-k") + 1] == "64"
    assert cmd[cmd.index("--manipulator-top-k") + 1] == "128"


def test_driver_honours_an_explicit_manipulator_top_k():
    args = _driver_args("--object-top-k", "32", "--manipulator-top-k", "96")
    assert args.object_top_k == 32
    assert args.manipulator_top_k == 96
    cmd = _build_3b_cmd(args, repo_root=__import__("pathlib").Path("/repo"))
    assert cmd[cmd.index("--object-top-k") + 1] == "32"
    assert cmd[cmd.index("--manipulator-top-k") + 1] == "96"


# --- Step 4: role caps + support auto-trim -----------------------------------


def test_role_max_keypoints_object_64_manipulator_128():
    assert ROLE_MAX_KEYPOINTS == {"object": 64, "manipulator": 128}


def test_role_keypoint_cap_unlabelled_role_is_64():
    assert _role_keypoint_cap(None) == 64
    assert _role_keypoint_cap("object") == 64
    assert _role_keypoint_cap("manipulator") == 128


def test_kept_keypoint_cap_is_64_per_prompt():
    """The filtered output keeps at most 64 keypoints per prompt (the
    object density) — a 128-seed manipulator prompt starts denser but its
    output never exceeds the old 64-keypoint schema."""
    assert MAX_KEPT_KEYPOINTS == 64


def test_cap_keypoint_survivors_leaves_64_or_fewer_untouched():
    sel = np.zeros(70, dtype=bool)
    sel[:50] = True                      # 50 survivors <= 64 -> untouched
    out, over = _cap_keypoint_survivors(sel, MAX_KEPT_KEYPOINTS)
    assert np.array_equal(out, sel)
    assert over.shape == (0,)


def test_cap_keypoint_survivors_truncates_to_first_64_survivors():
    # 128 seeds with 100 survivors (indices 0..19 and 30..109)
    sel = np.zeros(128, dtype=bool)
    sel[:20] = True
    sel[30:110] = True                   # 100 survivors in seed order
    out, over = _cap_keypoint_survivors(sel, MAX_KEPT_KEYPOINTS)
    assert out.sum() == 64
    # the first 64 survivors in seed order are kept
    expected = np.zeros(128, dtype=bool)
    expected[:20] = True
    expected[30:74] = True               # 20 + 44 = 64
    assert np.array_equal(out, expected)
    assert np.array_equal(over, np.arange(74, 110))  # capped-away positions
    # the caller's mask is not mutated
    assert sel.sum() == 100


def test_cap_keypoint_survivors_exactly_64_survivors_kept():
    sel = np.zeros(64, dtype=bool)
    sel[:] = True
    out, over = _cap_keypoint_survivors(sel, MAX_KEPT_KEYPOINTS)
    assert np.array_equal(out, sel)
    assert over.shape == (0,)


def test_support_queries_trims_grid_evenly_without_query_pixels():
    """Without query pixels the trim keeps the even linspace spread (the
    fallback of _support_queries)."""
    h, w = 48, 64
    depth = np.ones((h, w), dtype=np.float32)          # all-valid depth
    intrs = np.array([[100.0, 0.0, w / 2],
                      [0.0, 100.0, h / 2],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
    extr = np.eye(4, dtype=np.float32)
    rng = np.random.default_rng(0)
    need = 1088 - 128
    support = _support_queries(depth, intrs, extr, need, rng)
    assert support.shape == (need, 4)
    assert support.dtype == torch.float32
    full = _support_queries(depth, intrs, extr,
                            SUPPORT_GRID_SIZE ** 2, rng)
    assert full.shape == (1024, 4)
    pick = np.linspace(0, full.shape[0] - 1, need).round().astype(int)
    assert torch.equal(support, full[pick])


# --- query-aware support drop (pixel distance to the keypoints) ---------------


def _line_cells(n: int) -> np.ndarray:
    """n grid cells on a line: (i, 0) for i in 0..n-1."""
    cells = np.zeros((n, 2), dtype=np.float32)
    cells[:, 0] = np.arange(n, dtype=np.float32)
    return cells


def test_drop_cells_near_px_drops_each_queries_closest_cell():
    cells = _line_cells(10)
    queries = np.array([[1.2, 0.0], [4.2, 0.0], [8.2, 0.0]], np.float32)
    drop = _drop_cells_near_px(cells, queries, drop_budget=2)
    # budget 2 -> the first two queries shed their closest cell (cyclically)
    expected = np.zeros(10, dtype=bool)
    expected[[1, 4]] = True
    assert np.array_equal(drop, expected)


def test_drop_cells_near_px_skips_cells_already_dropped():
    """Two queries at the same pixel: the first claims the cell, the second
    falls back to its next closest."""
    cells = _line_cells(10)
    queries = np.array([[4.2, 0.0], [4.2, 0.0]], np.float32)
    drop = _drop_cells_near_px(cells, queries, drop_budget=2)
    expected = np.zeros(10, dtype=bool)
    expected[[4, 5]] = True            # cell 5 is the closest remaining to q1
    assert np.array_equal(drop, expected)


def test_drop_cells_near_px_cycles_when_budget_exceeds_queries():
    cells = _line_cells(10)
    queries = np.array([[4.2, 0.0], [6.2, 0.0]], np.float32)
    drop = _drop_cells_near_px(cells, queries, drop_budget=4)
    expected = np.zeros(10, dtype=bool)
    expected[[4, 6, 5, 7]] = True      # q0: 4, then 5 | q1: 6, then 7 (cyclic)
    assert np.array_equal(drop, expected)


def test_drop_cells_near_px_zero_budget_drops_nothing():
    cells = _line_cells(5)
    queries = np.array([[1.2, 0.0]], np.float32)
    drop = _drop_cells_near_px(cells, queries, drop_budget=0)
    assert not drop.any()


def test_support_queries_drops_cells_nearest_query_pixels():
    """End to end: a 128-keypoint pass (grid 1024, budget 64) drops the
    grid cells whose anchor-frame pixels are nearest the keypoints, keeps
    the rest in grid order."""
    h, w = 48, 64
    depth = np.ones((h, w), dtype=np.float32)          # all-valid depth
    intrs = np.array([[100.0, 0.0, w / 2],
                      [0.0, 100.0, h / 2],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
    extr = np.eye(4, dtype=np.float32)
    rng = np.random.default_rng(0)
    need = 1088 - 128
    full = _support_queries(depth, intrs, extr,
                            SUPPORT_GRID_SIZE ** 2, rng)
    assert full.shape == (1024, 4)
    # queries placed exactly on two grid cells (zero pixel distance)
    cell_px = get_points_on_a_grid(SUPPORT_GRID_SIZE,
                                   (h, w))[0].numpy()
    query_px = cell_px[[5, 500]]
    support = _support_queries(depth, intrs, extr, need, rng,
                               query_px=query_px)
    assert support.shape == (need, 4)
    # every support row is an exact grid row, in grid order (exact
    # equality — cdist tolerances are too coarse for these magnitudes)
    eq = (support[:, 1:].unsqueeze(1) == full[:, 1:].unsqueeze(0)).all(-1)
    ismem = eq.any(dim=0)                              # (1024,) rows kept
    assert ismem.sum() == need
    kept = ismem.nonzero().squeeze(1)
    assert (kept[1:] > kept[:-1]).all()                # order preserved
    assert torch.equal(support, full[kept])
    # the two query-positioned cells (distance 0) are among the dropped
    assert not ismem[5].item()
    assert not ismem[500].item()
    # a non-query cell survives
    assert ismem[700].item()
