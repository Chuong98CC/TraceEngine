"""Similarity registration and re-rendering of MoGe geometry.

Exercises ``utils.visualize.moge_register`` — the numpy port that aligns
per-frame MoGe geometry to an anchor frame. Pure numpy, so these run on CPU.

The central fixture is the unprojection/projection pair: MoGe's intrinsics are a
(3, 3) matrix in *normalized uv* units (``fx = k[0,0]``, ``cx = k[0,2] = 0.5``),
pixel centres sit at ``(j + 0.5) / W`` and ``(i + 0.5) / H``, and the model
returns ``X = (u - cx)/fx * Z``, so ``points[..., 2]`` is exactly the depth. The
projection below is the algebraic inverse, which is why an identity similarity
must round-trip bit for bit — most of the value of these tests is pinning that
invariant, since everything downstream inherits the convention.
"""

import inspect

import numpy as np
import pytest

from utils.visualize.moge_register import (
    Registration,
    register_to_anchor,
    reproject_to_camera,
    rigid_registration,
    weighted_mean_numpy,
)

#: Normalized-uv intrinsics. Deliberately off 1.0 and asymmetric between the
#: axes, so a transposed fx/fy or a swapped row/column would move the pixels.
FX, FY = 0.9, 1.1
CX, CY = 0.5, 0.5
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

#: Dyadic intrinsics (fx = fy = 1, so every projection is exact in binary
#: floating point) for the tests that need a *predicted* pixel or sub-pixel
#: offset rather than just a round trip.
K_EXACT = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])


def _rotation(axis, degrees):
    """Rotation matrix from a Rodrigues axis/angle, with the axis normalized."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(degrees)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _point_map(height=64, width=64, seed=7, z_range=(0.6, 3.0), dtype=np.float64):
    """Unproject a random depth map through `K`, as the model does.

    Returns ``(points, depth)``. The depth is deliberately spread over a range
    rather than constant: the registration weights by ``1 / depth``, so a
    constant map would hide any weighting mistake, and the translation gate is
    relative to the median depth.
    """
    j, i = np.meshgrid(np.arange(width), np.arange(height))
    u = (j + 0.5) / width
    v = (i + 0.5) / height
    depth = (
        z_range[0] + (z_range[1] - z_range[0]) * np.random.default_rng(seed).random((height, width))
    ).astype(dtype)
    # The uv grids are float64, so the stack has to be cast or the maps come back
    # in double precision and the float32 round trips below would compare an
    # already-rounded result against an exact one.
    points = np.stack([(u - CX) / FX * depth, (v - CY) / FY * depth, depth], axis=-1)
    return points.astype(dtype, copy=False), depth


# ---------------------------------------------------------------------------
# weighted_mean_numpy
# ---------------------------------------------------------------------------


def test_weighted_mean_without_weights_is_a_plain_mean():
    x = np.array([[1.0, 10.0], [3.0, 20.0]])
    assert np.allclose(weighted_mean_numpy(x, axis=0), [2.0, 15.0])
    assert np.allclose(weighted_mean_numpy(x, axis=-1), [5.5, 11.5])
    assert np.allclose(weighted_mean_numpy(x), 8.5)


def test_weighted_mean_matches_hand_computed_values():
    x = np.array([[1.0, 10.0], [3.0, 20.0]])
    # Row weights [3, 1] on 4 total weight: (3*1 + 1*3)/4 = 1.5 and
    # (3*10 + 1*20)/4 = 12.5.
    w = np.array([[3.0], [1.0]])
    assert np.allclose(weighted_mean_numpy(x, w, axis=0), [1.5, 12.5])


def test_weighted_mean_ignores_keepdims():
    """Upstream accepts `keepdims` but never forwards it — the axis is dropped.

    Callers rely on the upstream shape, so this is pinned rather than fixed.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([[3.0], [1.0]])
    assert weighted_mean_numpy(x, w, axis=0, keepdims=True).shape == (2,)
    assert weighted_mean_numpy(x, axis=0, keepdims=True).shape == (2,)


# ---------------------------------------------------------------------------
# rigid_registration
# ---------------------------------------------------------------------------


def test_rigid_registration_recovers_known_similarity():
    p = np.random.default_rng(0).normal(size=(200, 3)) * 0.5 + np.array([0.0, 0.0, 2.0])
    rotation = _rotation([0.2, 0.5, -0.8], 35.0)
    q = 1.7 * p @ rotation.T + np.array([0.3, -0.2, 0.05])

    scale, rotation_hat, translation_hat = rigid_registration(p, q)

    assert scale == pytest.approx(1.7, rel=1e-9)
    assert np.allclose(rotation_hat, rotation, atol=1e-9)
    assert np.allclose(translation_hat, [0.3, -0.2, 0.05], atol=1e-9)


def test_rigid_registration_rejects_a_reflection():
    """A mirrored correspondence set must not yield a mirrored "rotation".

    The determinant fix flips the third singular vector, so the returned matrix
    stays a proper rotation even though the best *orthogonal* fit to the data is
    a reflection (det = -1).
    """
    p = np.random.default_rng(1).normal(size=(200, 3)) * 0.5 + np.array([0.0, 0.0, 2.0])
    reflection = np.diag([1.0, 1.0, -1.0])
    q = p @ reflection.T

    _, rotation_hat, _ = rigid_registration(p, q)

    assert np.linalg.det(rotation_hat) > 0
    assert np.allclose(rotation_hat @ rotation_hat.T, np.eye(3), atol=1e-9)


# ---------------------------------------------------------------------------
# register_to_anchor
# ---------------------------------------------------------------------------


def test_register_to_anchor_recovers_known_similarity():
    points, depth = _point_map()
    rotation = _rotation([0.3, 0.8, 0.5], 2.5)
    anchor_points = 1.03 * points @ rotation.T + np.array([0.03, -0.02, 0.04])

    registration = register_to_anchor(points, depth, anchor_points, anchor_points[..., 2])

    assert isinstance(registration, Registration)
    assert registration.scale == pytest.approx(1.03, rel=1e-6)
    assert np.allclose(registration.rotation, rotation, atol=1e-6)
    assert np.allclose(registration.translation, [0.03, -0.02, 0.04], atol=1e-6)
    # An exact similarity leaves every sampled correspondence an inlier.
    assert registration.inliers > 0.95
    assert registration.n_points == 64 * 64


def test_register_to_anchor_subsamples_to_the_sample_budget():
    points, depth = _point_map()
    anchor_points = 1.01 * points

    registration = register_to_anchor(
        points,
        depth,
        anchor_points,
        anchor_points[..., 2],
        sample=500,
        hypothetical_size=200,
    )

    assert registration.n_points == 500
    assert registration.scale == pytest.approx(1.01, rel=1e-6)


def test_register_to_anchor_requires_common_valid_pixels():
    """Only pixels usable in *both* maps are correspondences.

    The anchor here is the same static view (identity-pixel correspondence), so
    the fit stays exact and the sampled count is exactly the number of common
    pixels — any pixel dropped by a NaN or a non-positive depth shows up as one
    fewer sample.
    """
    points, depth = _point_map()
    anchor_points, anchor_depth = points.copy(), depth.copy()

    # Five single pixels unusable in one map or the other: two NaN frame points,
    # a non-positive frame depth, a NaN anchor point and a negative anchor depth.
    points[0, 0] = np.nan
    points[5, 5] = np.nan
    depth[1, 1] = 0.0
    anchor_points[2, 2] = np.nan
    anchor_depth[3, 3] = -1.0

    registration = register_to_anchor(points, depth, anchor_points, anchor_depth)
    assert registration.n_points == 64 * 64 - 5

    # A blind half of the anchor takes that whole half out of the fit, and the
    # remaining half still registers.
    anchor_depth = anchor_depth.copy()
    anchor_depth[:, :32] = np.nan
    registration = register_to_anchor(points, depth, anchor_points, anchor_depth)
    assert registration.n_points == 32 * 64
    assert registration.inliers > 0.95


def test_register_to_anchor_none_when_too_few_common_pixels():
    """Fewer common pixels than one RANSAC hypothesis cannot even be sampled.

    The hypothesis size is passed explicitly rather than left to the default:
    the gate is `n_common < hypothetical_size`, so a test that relied on the
    default would stop exercising the gate the moment the default moved -- and
    it has moved once already, from 2000 to the minimal-sample 10.
    """
    points, depth = _point_map(height=10, width=10)  # 100 pixels, all common
    assert register_to_anchor(
        points, depth, points, depth, hypothetical_size=200
    ) is None

    # And it is accepted as soon as the hypothesis fits inside the overlap,
    # which is what makes the assertion above about the gate rather than about
    # the map.
    assert register_to_anchor(
        points, depth, points, depth, hypothetical_size=50
    ) is not None


def test_register_to_anchor_none_for_fewer_than_three_points():
    """Two points fix a similarity only up to a rotation about their axis."""
    points, depth = _point_map(height=1, width=2)
    registration = register_to_anchor(
        points, depth, points, depth, hypothetical_size=2, max_iters=4
    )
    assert registration is None


def test_register_to_anchor_none_for_unrelated_anchor():
    """A frame from another camera must be refused, not aligned to anything.

    The anchor is drawn from the same scene statistics as the frame but shares
    no correspondence with it — the case where a plausible-looking transform
    would be pure invention.
    """
    points, depth = _point_map()
    anchor_points, anchor_depth = _point_map(seed=11)

    assert register_to_anchor(points, depth, anchor_points, anchor_depth) is None


def _partly_moving_pair():
    """A frame whose far half moved and whose near half did not.

    The transform recovered from the static half is the near-identity a fixed
    camera calls for, so nothing but the inlier count can tell this frame apart
    from a good one. It scores 0.667 — measured, and the reason the gate below
    has to be passed explicitly.
    """
    points, depth = _point_map()
    anchor_points = points.copy()
    anchor_points[depth > 2.2] += np.array([0.4, 0.3, 0.2])
    return points, depth, anchor_points


def test_register_to_anchor_none_when_too_few_inliers():
    """The inlier gate is what refuses a partly-moving camera, if it is set high."""
    points, depth, anchor_points = _partly_moving_pair()

    assert register_to_anchor(
        points, depth, anchor_points, anchor_points[..., 2], min_inliers=0.8
    ) is None

    # Lifting the gate shows why: the transform was plausible all along, so no
    # other gate would have caught this frame.
    permissive = register_to_anchor(
        points, depth, anchor_points, anchor_points[..., 2], min_inliers=0.0
    )
    assert permissive.inliers < 0.8
    assert permissive.scale == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(permissive.rotation, np.eye(3), atol=1e-6)
    assert np.allclose(permissive.translation, np.zeros(3), atol=1e-6)


def test_the_default_gate_deliberately_admits_a_partly_moving_camera():
    """The trade the default `min_inliers` makes, pinned so it is not a surprise.

    The same frame as above is ACCEPTED by the default, and that is intended
    rather than a slipped threshold: the default has to sit below the 64-71%
    that a legitimate long-gap anchor registration reads, and this frame's 0.667
    is inside that band. The two populations overlap, so no per-frame threshold
    separates a fixed camera from a moving one -- which is why the caller
    declares which is which, and why `register_to_anchor` cannot be the thing
    that decides.
    """
    points, depth, anchor_points = _partly_moving_pair()
    registration = register_to_anchor(points, depth, anchor_points,
                                      anchor_points[..., 2])
    default = inspect.signature(register_to_anchor).parameters["min_inliers"].default
    assert default < 0.667, (
        "the default has risen above a partly-moving frame's inlier ratio, so "
        "this frame would now be refused and this test no longer documents the "
        "trade it was written for"
    )
    assert registration is not None
    assert registration.inliers == pytest.approx(0.667, abs=0.01)


def test_register_to_anchor_accepts_a_large_but_real_drift():
    """A transform far from the identity is exactly what this is here to fit.

    The model's depth scale is re-estimated every frame, so over a long episode
    a frame's geometry genuinely differs from its anchor's by more than a
    little: measured over three LIBERO episodes the fitted scale runs 0.97 to
    1.46 at 45-73% inliers, and every one of those is correct. A gate that
    reads "the transform should be near the identity" therefore rejects the
    registrations that matter most -- which is what a `max_scale_error` of 0.1
    did, refusing two episodes out of three outright.
    """
    points, depth = _point_map()
    for factor in (0.97, 1.2, 1.46):
        anchor_points = factor * points
        registration = register_to_anchor(
            points, depth, anchor_points, anchor_points[..., 2]
        )
        assert registration is not None, f"scale {factor} was refused"
        assert registration.scale == pytest.approx(factor, abs=1e-6)


def test_register_to_anchor_rejects_an_absurd_scale():
    """The rail still exists; it is just no longer a discriminator."""
    points, depth = _point_map()
    anchor_points = 2.0 * points  # s = 2.0, past the 1.75 rail

    assert register_to_anchor(points, depth, anchor_points, anchor_points[..., 2]) is None


def test_register_to_anchor_rejects_implausible_rotation():
    points, depth = _point_map()
    anchor_points = points @ _rotation([0.0, 0.0, 1.0], 20.0).T

    assert register_to_anchor(points, depth, anchor_points, anchor_points[..., 2]) is None


def test_register_to_anchor_rejects_implausible_translation():
    points, depth = _point_map()
    # Median anchor depth is about 2.29 m, so the gate sits near 0.57 m.
    anchor_points = points + np.array([0.0, 0.0, 1.0])

    assert register_to_anchor(points, depth, anchor_points, anchor_points[..., 2]) is None

    # ... and a translation inside the gate is accepted, so the gate above is a
    # threshold rather than a blanket refusal.
    anchor_points = points + np.array([0.05, 0.0, 0.05])
    registration = register_to_anchor(points, depth, anchor_points, anchor_points[..., 2])
    assert registration is not None
    assert np.allclose(registration.translation, [0.05, 0.0, 0.05], atol=1e-6)


def test_register_to_anchor_translation_gate_is_relative_to_the_scene():
    """The gate is a fraction of the anchor's depth scale, not a fixed distance.

    A 25 cm wobble is plausible for a camera fixed in a room (median depth
    around 3 m, so the gate sits near 30 cm) and wildly implausible in a scene a
    tenth that size. An absolute metre threshold would have to get one of those
    two wrong, and the monocular scale is only ever defined up to the estimate —
    so nothing absolute can be meaningful here.
    """
    translation = np.array([0.0, 0.0, 0.25])

    room_points, room_depth = _point_map(z_range=(1.0, 5.0))
    room_anchor = room_points + translation
    assert register_to_anchor(room_points, room_depth, room_anchor, room_anchor[..., 2]) is not None

    # The same shift on a scene an order of magnitude nearer is 25% of its
    # median depth, and the gate closes.
    close_points, close_depth = _point_map(z_range=(0.06, 0.3))
    close_anchor = close_points + translation
    assert register_to_anchor(close_points, close_depth, close_anchor, close_anchor[..., 2]) is None


# ---------------------------------------------------------------------------
# reproject_to_camera
# ---------------------------------------------------------------------------


def test_reproject_identity_is_bit_exact():
    """The oracle: identity + same intrinsics + same size reproduces the maps.

    The projection is the algebraic inverse of the unprojection, so a
    re-render that differs at all means the convention is wrong somewhere —
    which is exactly the failure this pins down.
    """
    height, width = 7, 11  # non-square, so a transposed axis shows up
    points, depth = _point_map(height=height, width=width, dtype=np.float32)
    assert np.array_equal(points[..., 2], depth), "fixture must satisfy points[..., 2] == depth"
    normals = np.random.default_rng(5).normal(size=(height, width, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = np.ones((height, width), dtype=bool)

    out_depth, out_normals, out_valid, hole_fraction = reproject_to_camera(
        points, normals, valid, 1.0, np.eye(3), np.zeros(3), K, (height, width)
    )

    assert np.array_equal(out_depth, depth)
    assert np.array_equal(out_normals, normals)
    assert np.array_equal(out_valid, valid)
    assert hole_fraction == 0.0


def test_reproject_keeps_invalid_pixels_as_holes():
    height, width = 7, 11
    points, depth = _point_map(height=height, width=width, dtype=np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    valid = np.ones((height, width), dtype=bool)
    valid[0, 0] = False
    valid[3, :] = False

    out_depth, _, out_valid, hole_fraction = reproject_to_camera(
        points, normals, valid, 1.0, np.eye(3), np.zeros(3), K, (height, width)
    )

    # Invalid source pixels are holes (not filled), and they do not disturb the
    # pixels that did land.
    assert np.array_equal(out_valid, valid)
    assert hole_fraction == pytest.approx(1.0 - valid.mean())
    assert np.array_equal(out_depth[valid], depth[valid])


def test_reproject_drops_non_finite_points():
    height, width = 7, 11
    points, depth = _point_map(height=height, width=width, dtype=np.float32)
    points = points.copy()
    points[2, 3] = np.nan
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    out_depth, out_normals, out_valid, hole_fraction = reproject_to_camera(
        points, normals, np.ones((height, width), bool), 1.0, np.eye(3), np.zeros(3), K,
        (height, width),
    )

    assert not out_valid[2, 3]
    assert out_depth[2, 3] == 0.0
    assert np.array_equal(out_normals[2, 3], np.zeros(3))
    assert hole_fraction == pytest.approx(1.0 / (height * width))
    # Everything else still round-trips.
    assert np.array_equal(out_depth[out_valid], depth[out_valid])


def test_reproject_drops_points_behind_the_camera():
    """A negative Z is not a point, it is a ray in the opposite direction.

    Testing for ``Z > 0`` and not merely for finiteness matters: the projection
    of a negative Z mirrors through the image centre, and being the *nearest*
    thing to the camera it would win the z-buffer wherever it landed.
    """
    height, width = 7, 11
    points, depth = _point_map(height=height, width=width, dtype=np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    points = points.copy()
    points[4, 5] = np.array([0.1, 0.1, -1.0], dtype=np.float32)

    out_depth, out_normals, out_valid, hole_fraction = reproject_to_camera(
        points, normals, np.ones((height, width), bool), 1.0, np.eye(3), np.zeros(3), K,
        (height, width),
    )

    assert not out_valid[4, 5]
    assert hole_fraction == pytest.approx(1.0 / (height * width))
    # Where it would have landed, the point that really lives there survives.
    assert out_depth[2, 4] == depth[2, 4]
    assert np.array_equal(out_normals[2, 4], normals[2, 4])
    assert (out_depth[out_valid] > 0).all()


def test_reproject_translation_moves_the_frame_by_the_predicted_pixels():
    """On a constant-depth frame a translation is a pure, exact pixel shift."""
    height, width = 2, 8
    points, depth = _point_map(
        height=height, width=width, seed=2, z_range=(1.0, 1.0), dtype=np.float32
    )
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    # fx = 1, Z = 1, so tx = 3 / W shifts the image right by exactly 3 pixels.
    translation = np.array([3.0 / width, 0.0, 0.0])

    out_depth, _, out_valid, hole_fraction = reproject_to_camera(
        points, normals, np.ones((height, width), bool), 1.0, np.eye(3), translation,
        K_EXACT, (height, width),
    )

    expected = np.zeros((height, width), dtype=bool)
    expected[:, 3:] = True
    assert np.array_equal(out_valid, expected)
    assert np.array_equal(out_depth[:, 3:], depth[:, :-3])
    # The three columns pushed off the left edge are reported as holes, not filled.
    assert hole_fraction == pytest.approx(3.0 / width)


def test_reproject_scale_scales_the_depths():
    height, width = 7, 11
    points, depth = _point_map(height=height, width=width, dtype=np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    out_depth, out_normals, out_valid, hole_fraction = reproject_to_camera(
        points, normals, np.ones((height, width), bool), 2.5, np.eye(3), np.zeros(3), K,
        (height, width),
    )

    # Scaling about the camera centre leaves each point on its own ray, so the
    # pixel is unchanged and only the depth moves.
    assert np.array_equal(out_valid, np.ones((height, width), bool))
    assert hole_fraction == 0.0
    assert np.array_equal(out_depth, (2.5 * depth).astype(np.float32))
    assert np.array_equal(out_normals, normals)


def test_reproject_takes_depth_and_normal_from_one_source_pixel():
    """A target pixel's depth and normal must describe the same surface.

    There is no z-buffer to enforce this -- the map is sampled backwards rather
    than splatted forwards, so no two source pixels ever compete for a target
    pixel -- but the property still has to hold, and it now rests on both
    channels being read at the single source index the inverse map produced.

    Each source pixel's normal is tagged with its own column, so a depth taken
    from one source pixel and a normal from another cannot agree.
    """
    width = 8
    points, _ = _point_map(height=1, width=width)
    tagged = np.zeros((1, width, 3), dtype=np.float32)
    tagged[0, :, 0] = np.arange(width)  # the tag, read back as the source column
    tagged[0, :, 2] = 1.0
    translation = np.array([0.05, 0.0, 0.1])

    out_depth, out_normals, out_valid, _ = reproject_to_camera(
        points, tagged, np.ones((1, width), bool), 1.0, np.eye(3), translation,
        K, (1, width),
    )

    assert out_valid.any(), "the fixture must keep some pixels to test with"
    for col in np.flatnonzero(out_valid[0]):
        # The identity rotation passes the tag through untouched, so the normal
        # names the source pixel the depth has to have come from.
        source_col = int(round(float(out_normals[0, col, 0])))
        assert out_depth[0, col] == pytest.approx(
            points[0, source_col, 2] + translation[2]
        ), f"column {col} mixed depth and normal from different source pixels"


def test_reproject_a_rotation_leaves_no_lattice_of_holes():
    """The artifact a forward splat produces, pinned.

    Rounding every source point onto the nearest target pixel does not cover a
    rotated lattice: a rotation lands the source grid off the target grid and a
    regular picket of pixels gets nothing written to it, which on real frames
    is plainly visible as a fine grid over the whole image. Sampling backwards
    cannot do that -- every target pixel is visited exactly once -- so the only
    holes left are where the inverse map leaves the frame, and the interior
    must be complete.
    """
    height = width = 64
    points, _ = _point_map(height=height, width=width)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    angle = np.radians(1.0)
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(angle), 0.0, np.cos(angle)]])

    _, _, out_valid, hole_fraction = reproject_to_camera(
        points, normals, np.ones((height, width), bool), 1.0, rotation,
        np.zeros(3), K, (height, width),
    )

    margin = 8  # the rotation can push the frame's own edge out of view
    interior = out_valid[margin:-margin, margin:-margin]
    assert interior.all(), (
        f"{int((~interior).sum())} hole(s) in the interior: that is the lattice "
        f"a forward splat leaves, not edge loss"
    )
    assert hole_fraction < 0.05, f"{hole_fraction:.1%} holes is edge loss and more"


def test_reproject_rotates_normals_without_scaling_or_translating_them():
    """Normals are directions: a similarity rotates them and nothing else.

    One on-axis point in a one-pixel frame, so the projection lands whatever the
    rotation and translation do. The expected normal is derived by hand from the
    rotation matrix — a normal treated as a point would additionally be scaled
    (making it non-unit) and shifted by the translation, so the two calls below,
    which differ only in their translation, must agree exactly.
    """
    points = np.array([[[0.0, 0.0, 2.0]]], dtype=np.float32)
    normals = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    rotation = _rotation([1.0, 0.0, 0.0], 10.0)
    angle = np.radians(10.0)
    expected = np.array([0.0, -np.sin(angle), np.cos(angle)])

    _, out_normals, out_valid, _ = reproject_to_camera(
        points, normals, np.ones((1, 1), bool), 2.0, rotation, np.zeros(3), K, (1, 1)
    )
    assert out_valid[0, 0]
    assert np.allclose(out_normals[0, 0], expected, atol=1e-6)
    # Unit length: a scale of 2.0 applied to the normal would give 2.0 here.
    assert np.isclose(np.linalg.norm(out_normals[0, 0]), 1.0, atol=1e-6)

    # ... and translating the point must leave the normal bit-identical.
    _, translated_normals, translated_valid, _ = reproject_to_camera(
        points,
        normals,
        np.ones((1, 1), bool),
        2.0,
        rotation,
        np.array([0.0, 0.0, 0.5]),
        K,
        (1, 1),
    )
    assert translated_valid[0, 0]
    assert np.array_equal(translated_normals[0, 0], out_normals[0, 0])
