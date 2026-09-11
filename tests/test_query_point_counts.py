"""CPU-only unit tests for the role keypoint/query budget.

The static filter of Step 4 (--filter-static-pixel) removes the
stationary keypoints, so the role passes need many seeds to keep enough
moving survivors. The count is capped in two places that must move
together:

- Step 3b (tools/general_test/pipeline/run_object_init_points.py)
  samples at most ``--object-top-k`` keypoints per prompt into
  init_points.npz; ``--manipulator-top-k`` raises the cap for the
  manipulator-role prompts only (object and role-less prompts keep
  ``--object-top-k``).
- Step 4 (tools/astribot/run_step4_traces.py) caps the role keypoints of
  each pass at ROLE_MAX_KEYPOINTS[role] (1000 per role — Step 4 roles
  are plain object/manipulator, an unmatched prompt is skipped).
  The shipped TAPIP3D iteration graph is fixed at EXPECTED_NUM_QUERIES
  (1088), so the support block is whatever the role keypoints leave over
  (88 for a role pass at its full 1000-keypoint cap): the anchor frame's
  valid-depth pixels that no role prompt's mask covers are drawn at
  random with a separable Gaussian weight peaking at the image centre
  (sigma = --support-sigma times each image dimension) — see
  ``_support_queries``. The draw is without replacement whenever the
  eligible pool is at least the budget (distinct support pixels), and
  falls back to drawing WITH replacement (a warning) for a pool too
  small to fill the exact-N budget, rather than raising; a
  non-positive ``--support-sigma`` is rejected by the parser.

``_role_mask_union`` builds that exclusion mask — the union of every role
prompt's mask at the anchor frame, column-picked like the keypoints.

The Step-3 driver (tools/astribot/run_step3_init_points.py) forwards
``--object-top-k`` and ``--manipulator-top-k`` (default 128) to Step 3b.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.astribot.run_step3_init_points import (
    DEFAULT_MANIPULATOR_TOP_K,
    DEFAULT_OBJECT_TOP_K,
    _build_3b_cmd,
    parse_args,
)
from tools.astribot.run_step4_traces import (
    DEFAULT_SUPPORT_SIGMA,
    EXPECTED_NUM_QUERIES,
    ROLE_MAX_KEYPOINTS,
    ROLE_ORDER,
    _role_keypoint_cap,
    _role_mask_union,
    _support_queries,
    parse_args as step4_parse_args,
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


def _step4_args(*extra: str):
    """Step-4 parse_args, the same way (its --repo-id/--data-root are
    required, so they are always supplied)."""
    return step4_parse_args(["--repo-id", "r", "--data-root", "/tmp/x",
                             *extra])


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


# --- Step 4: role caps -------------------------------------------------------


def test_role_max_keypoints_is_1000_per_role():
    assert ROLE_MAX_KEYPOINTS == {"object": 1000, "manipulator": 1000}


def test_role_keypoint_cap_is_the_roles_max_keypoints():
    for role in ROLE_ORDER:
        assert _role_keypoint_cap(role) == ROLE_MAX_KEYPOINTS[role]


# --- Step 4: support-query draw ----------------------------------------------
#
# Synthetic anchor geometry: an off-centre principal point, non-square
# intrinsics, and a rotation + translation extrinsics (a world->camera
# pose that is neither identity nor axis-aligned, so no round-trip
# assertion can pass on an identity shortcut). The tests reproject the
# returned world queries back to pixels with the exact inverse of
# unproject_xy_queries, so every assertion is in pixels on the depth grid.

H, W = 160, 200
INTRS = np.array([[520.0, 0.0, 68.0],
                  [0.0, 505.0, 41.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
_ANG = np.deg2rad(20.0)
EXTR = np.array([[1.0, 0.0, 0.0, 0.12],
                 [0.0, np.cos(_ANG), -np.sin(_ANG), -0.07],
                 [0.0, np.sin(_ANG), np.cos(_ANG), 0.35],
                 [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)


def _reproject(queries) -> np.ndarray:
    """(N, 2) anchor-frame pixels of the unprojected world queries — the
    exact inverse of unproject_xy_queries, so a query's drawn pixel on the
    depth grid is recoverable for assertions."""
    world = queries[:, 1:].numpy().astype(np.float64)
    homo = np.concatenate([world, np.ones((len(world), 1))], axis=-1)
    cam = (EXTR.astype(np.float64) @ homo.T).T[:, :3]
    fx, fy = float(INTRS[0, 0]), float(INTRS[1, 1])
    cx, cy = float(INTRS[0, 2]), float(INTRS[1, 2])
    return np.stack([cam[:, 0] / cam[:, 2] * fx + cx,
                     cam[:, 1] / cam[:, 2] * fy + cy], axis=-1)


def _pixels(queries) -> np.ndarray:
    """(N, 2) int pixels of the world queries on the depth grid."""
    return np.round(_reproject(queries)).astype(np.int64)


def _flat_depth(h: int = H, w: int = W, value: float = 1.5) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float32)


def test_role_caps_leave_a_positive_support_budget():
    """The graph's 1088 queries are fixed, so the passes' role caps must
    leave room for the support draw."""
    assert EXPECTED_NUM_QUERIES == 1088
    assert EXPECTED_NUM_QUERIES - max(ROLE_MAX_KEYPOINTS.values()) > 0


def test_support_sigma_must_be_positive():
    """0 divides by zero in the Gaussian and zeroes every weight, so the
    draw would report "no eligible support pixels" — blaming the data for
    a bad flag; a negative sigma would silently behave like a positive
    one. Both exit instead of running."""
    assert _step4_args().support_sigma == DEFAULT_SUPPORT_SIGMA
    assert _step4_args("--support-sigma", "0.5").support_sigma == 0.5
    with pytest.raises(SystemExit):
        _step4_args("--support-sigma", "0")
    with pytest.raises(SystemExit):
        _step4_args("--support-sigma", "-0.25")


def test_support_queries_returns_exactly_n_points_on_valid_depth():
    depth = _flat_depth()
    depth[0, :] = 0.0                       # a depth-invalid band
    depth[:, :10] = 0.0
    got = _support_queries(depth, INTRS, EXTR, 64,
                           np.random.default_rng(0), DEFAULT_SUPPORT_SIGMA)
    assert got.shape == (64, 4)
    assert got.dtype == torch.float32
    assert got.device.type == "cpu"
    assert (got[:, 0] == 0).all()                       # home frame 0
    px = _pixels(got)
    assert ((px[:, 0] >= 0) & (px[:, 0] < W)).all()
    assert ((px[:, 1] >= 0) & (px[:, 1] < H)).all()
    assert (depth[px[:, 1], px[:, 0]] > 0).all()      # valid depth only


def test_support_queries_draws_distinct_pixels_without_replacement():
    """replace=False by default: no pixel is drawn twice, so the support
    block can never duplicate a query (or a tracked keypoint's pixel).

    The pool is deliberately tiny (12x12 = 144 pixels) and the request
    close to it (128): over a pool of tens of thousands a replace=True
    implementation still yields all-distinct pixels at this seed, so the
    assertion below would pass on one. Here a replace=True draw repeats a
    pixel with overwhelming probability (all-distinct chance ~1e-40)."""
    got = _support_queries(_flat_depth(h=12, w=12), INTRS, EXTR, 128,
                           np.random.default_rng(1), DEFAULT_SUPPORT_SIGMA)
    px = _pixels(got)
    assert len(px) == 128
    assert len({(int(x), int(y)) for x, y in px}) == 128


def test_support_queries_never_lands_inside_the_exclusion_mask():
    depth = _flat_depth()
    depth[100:, :] = 0.0                    # the lower band: no depth
    exclude = np.zeros((H, W), dtype=bool)
    exclude[20:90, 30:120] = True           # the role masks' union
    got = _support_queries(depth, INTRS, EXTR, 128,
                           np.random.default_rng(2), DEFAULT_SUPPORT_SIGMA,
                           exclude=exclude)
    px = _pixels(got)
    assert len(px) == 128
    assert not exclude[px[:, 1], px[:, 0]].any()
    assert (depth[px[:, 1], px[:, 0]] > 0).all()


def test_support_queries_are_centre_weighted():
    """The Gaussian weight peaks at the image centre: the draw's mean
    distance from the centre is clearly below a uniform draw's over the
    same grid (both seeded, so the comparison is exact)."""
    n = 512
    got = _support_queries(_flat_depth(), INTRS, EXTR, n,
                           np.random.default_rng(3), DEFAULT_SUPPORT_SIGMA)
    px = _pixels(got)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    d_draw = np.hypot(px[:, 0] - cx, px[:, 1] - cy)
    ys, xs = np.nonzero(_flat_depth() > 0)
    pick = np.random.default_rng(3).choice(xs.size, size=n, replace=False)
    d_uniform = np.hypot(xs[pick] - cx, ys[pick] - cy)
    assert d_draw.mean() < 0.8 * d_uniform.mean()
    assert d_draw.max() < 2.5 * d_uniform.mean()      # spread, not a blob


def test_support_queries_are_deterministic():
    depth = _flat_depth()

    def draw(seed: int):
        return _support_queries(depth, INTRS, EXTR, 32,
                                np.random.default_rng(seed),
                                DEFAULT_SUPPORT_SIGMA)

    assert draw(7).equal(draw(7))
    assert not draw(7).equal(draw(8))


def test_support_queries_pad_with_replacement_when_the_pool_is_short(capsys):
    """A pool smaller than the graph's exact-N budget is drawn with
    replacement (a warning), never a raise — the graph must still receive
    exactly ``n_points`` queries."""
    depth = np.zeros((H, W), dtype=np.float32)
    pool_px = [(11, 5), (12, 5), (13, 5), (40, 60)]
    for x, y in pool_px:
        depth[y, x] = 1.2
    got = _support_queries(depth, INTRS, EXTR, 16,
                           np.random.default_rng(4), DEFAULT_SUPPORT_SIGMA)
    assert got.shape == (16, 4)
    px = _pixels(got)
    assert {tuple(p) for p in px} <= set(pool_px)
    assert "replacement" in capsys.readouterr().out


def test_support_queries_without_a_pool_raises():
    """No eligible pixel at all: no valid depth, or a mask covering every
    valid-depth pixel — both are a hard error (nothing to draw)."""
    with pytest.raises(ValueError):
        _support_queries(np.zeros((H, W), dtype=np.float32), INTRS, EXTR, 8,
                         np.random.default_rng(5), DEFAULT_SUPPORT_SIGMA)
    with pytest.raises(ValueError):
        _support_queries(_flat_depth(), INTRS, EXTR, 8,
                         np.random.default_rng(5), DEFAULT_SUPPORT_SIGMA,
                         exclude=np.ones((H, W), dtype=bool))


# --- Step 4: role-mask exclusion (_role_mask_union) ---------------------------


def _mask(shape=(H, W), x0=0, y0=0, x1=0, y1=0) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _prompt(masks: np.ndarray, frame_indices, prompt: str = "brown cup",
            slug: str = "brown_cup") -> dict:
    """Minimal Step-3b prompt dict: _role_mask_union only reads the masks
    (shape (N, H, W)) and the key-frames (N,)."""
    return {"masks": masks, "frame_indices": np.asarray(frame_indices),
            "prompt": prompt, "slug": slug}


def test_role_mask_union_is_none_without_a_mask():
    prompts = [_prompt(np.zeros((2, H, W), dtype=bool), [10, 20]),
               _prompt(np.zeros((2, H, W), dtype=bool), [20, 30])]
    assert _role_mask_union(prompts, 20, (H, W)) is None
    assert _role_mask_union([], 20, (H, W)) is None


def test_role_mask_union_unions_every_prompts_mask():
    a = _mask(x0=10, y0=10, x1=40, y1=40)
    b = _mask(x0=100, y0=60, x1=150, y1=120)
    prompts = [_prompt(np.stack([a, a]), [20, 30], prompt="brown cup"),
               _prompt(np.stack([b, b]), [20, 30],
                       prompt="robot arm's black grippers",
                       slug="robot_arm_s_black_grippers")]
    union = _role_mask_union(prompts, 20, (H, W))
    assert union.shape == (H, W)
    assert union.dtype == bool
    assert np.array_equal(union, a | b)


def test_role_mask_union_picks_the_anchors_column_when_it_is_a_key_frame():
    """The same column rule _usable_at uses for the keypoints: the anchor's
    own key-frame column when the anchor is one of the prompt's key-frames,
    the first column otherwise (the window's leading stem, at-or-before the
    first key-frame)."""
    m0 = _mask(x0=10, y0=10, x1=40, y1=40)
    m2 = _mask(x0=100, y0=60, x1=150, y1=120)
    prompts = [_prompt(np.stack([m0, _mask(), m2]), [10, 20, 30])]
    assert np.array_equal(_role_mask_union(prompts, 30, (H, W)), m2)
    assert np.array_equal(_role_mask_union(prompts, 15, (H, W)), m0)
    assert _role_mask_union(prompts, 20, (H, W)) is None    # empty column


def test_role_mask_union_nearest_resizes_a_key_frame_resolution_mask():
    """Masks are stored at the RGB key-frame resolution, which can differ
    from the depth grid (here 4x: 40x50 key-frames vs the 160x200 depth):
    the mask is nearest-resized before the union."""
    kf_mask = _mask(shape=(40, 50), x0=10, y0=10, x1=20, y1=20)
    prompts = [_prompt(kf_mask[None], [10])]
    union = _role_mask_union(prompts, 10, (H, W))
    expected = np.zeros((H, W), dtype=bool)
    expected[40:80, 40:80] = True             # the box's exact 4x caps
    assert union.shape == (H, W)
    assert np.array_equal(union, expected)


def test_role_mask_union_skips_all_false_columns():
    a = _mask(x0=10, y0=10, x1=40, y1=40)
    prompts = [_prompt(np.stack([_mask(), _mask()]), [10, 20]),
               _prompt(np.stack([a, a]), [10, 20])]
    assert np.array_equal(_role_mask_union(prompts, 10, (H, W)), a)
    assert np.array_equal(_role_mask_union(prompts, 20, (H, W)), a)
