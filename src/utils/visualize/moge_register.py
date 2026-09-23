#
# The rigid transformation here includes scale, rotation and translation. The raw output scale of MoGe is unconstrained and not consistent across video frames, since it has been trained to be scale-invariant for single images.
# Our implementation for RANSAC rigid (similarity) registration is quite simple.
# p: the current frame camera-space point;
# q: matched reference frame world-space point.
# w: inversely proportional to its depth.

from dataclasses import dataclass

import numpy as np


def weighted_mean_numpy(
    x: np.ndarray,
    w: np.ndarray = None,
    axis=None,
    keepdims: bool = False,
    eps: float = 1e-7,
) -> np.ndarray:
    """Weighted mean of ``x`` along ``axis``, with weights ``w``.

    Reproduced verbatim from MoGe's ``moge/utils/geometry_numpy.py``: this repo
    vendors only the torch half of that module
    (``depth_models/moge3/utils/geometry_torch.py``), and the registration code
    below is a numpy port that must run without torch. Upstream accepts
    ``keepdims`` but never forwards it to the two ``mean`` calls, so the reduced
    axis is always dropped; that quirk is preserved rather than fixed, because
    callers rely on the upstream shape.
    """
    if w is None:
        return np.mean(x, axis=axis)
    else:
        w = w.astype(x.dtype)
        return (x * w).mean(axis=axis) / np.clip(w.mean(axis=axis), eps, None)


def rigid_registration(
    p: np.ndarray,
    q: np.ndarray,
    w: np.ndarray = None,
    eps: float = 1e-12
) -> tuple[float, np.ndarray, np.ndarray]:
    if w is None:
        w = np.ones(p.shape[0])
    centroid_p = weighted_mean_numpy(p, w[:, None], axis=0)
    centroid_q = weighted_mean_numpy(q, w[:, None], axis=0)

    p_centered = p - centroid_p
    q_centered = q - centroid_q
    w = w / (np.sum(w) + eps)

    cov = (w[:, None] * p_centered).T @ q_centered
    U, S, Vh = np.linalg.svd(cov)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2, :] *= -1
        R = Vh.T @ U.T
    scale = np.sum(S) / np.trace((w[:, None] * p_centered).T @ p_centered)
    t = centroid_q - scale * (centroid_p @ R.T)
    return scale, R, t


def rigid_registration_ransac(
    p: np.ndarray,
    q: np.ndarray,
    w: np.ndarray = None,
    max_iters: int = 20,
    hypothetical_size: int = 10,
    inlier_thresh: float = 0.02
) -> tuple[float, np.ndarray, np.ndarray]:
    n = p.shape[0]
    if w is None:
        w = np.ones(p.shape[0])

    best_score, best_inlines = 0., np.zeros(n, dtype=bool)
    best_solution = (np.array(1.), np.eye(3), np.zeros(3))

    for _ in range(max_iters):
        maybe_inliers = np.random.choice(n, size=hypothetical_size, replace=False)
        try:
            s, R, t = rigid_registration(p[maybe_inliers], q[maybe_inliers], w[maybe_inliers])
        except np.linalg.LinAlgError:
            continue
        transformed_p = s * p @ R.T + t
        errors = w * np.linalg.norm(transformed_p - q, axis=1)
        inliers = errors < inlier_thresh

        score = inlier_thresh * n - np.clip(errors, None, inlier_thresh).sum()
        if  score > best_score:
            best_score, best_inlines = score, inliers
            best_solution = rigid_registration(p[inliers], q[inliers], w[inliers])

    return best_solution, best_inlines


#: RANSAC seed used when `register_to_anchor` is not handed a generator. Fixed
#: so that repeated calls on the same pair of maps subsample the same pixels;
#: note that only the subsampling is seeded — `rigid_registration_ransac` draws
#: its hypotheses from the global `np.random` state, so the recovered transform
#: can still wobble by a few ulps between runs.
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Registration:
    """A similarity from a frame's camera space into an anchor's world frame."""

    scale: float
    rotation: np.ndarray      # (3, 3), world-from-camera
    translation: np.ndarray   # (3,)
    inliers: float            # fraction of the sampled correspondences that agreed, in [0, 1]
    n_points: int             # how many correspondences were sampled


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, in degrees.

    `arccos` of the normalized trace is the standard closed form. The clip only
    guards the domain: an estimated matrix can push the trace a few ulps beyond
    ±1, which would make `arccos` return NaN and silently pass a rotation gate.
    """
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))))


def register_to_anchor(
    points: np.ndarray,
    depth: np.ndarray,
    anchor_points: np.ndarray,
    anchor_depth: np.ndarray,
    *,
    max_iters: int = 200,
    hypothetical_size: int = 10,
    inlier_thresh: float = 0.02,
    sample: int = 20000,
    min_inliers: float = 0.55,
    max_scale_error: float = 0.75,
    max_rotation_deg: float = 10.0,
    max_translation_frac: float = 0.25,
    rng: np.random.Generator | None = None,
) -> Registration | None:
    """Register one frame's camera geometry to an anchor frame, or reject it.

    `points`/`depth` are the (H, W, 3) camera-space points and (H, W) depth of
    the frame being registered; `anchor_points`/`anchor_depth` are the anchor's,
    in the anchor's own frame, which defines the world. Correspondence is
    **identity pixel** — pixel i of the frame is matched to pixel i of the
    anchor. That is only meaningful when the camera has not moved relative to
    the scene, so this is deliberately not a feature matcher: it exists to
    align frames of a *fixed* camera whose monocular depth is scale-ambiguous
    (the scale is re-estimated per frame), and to *refuse* frames where the
    camera clearly moved.

    Measured on the LIBERO episodes, and the reason `min_inliers` is as low as
    it is: the ratio depends far more on how many frames apart the pair is than
    on anything else. Consecutive frames of a fixed camera give 87-98%, but the
    model's geometry drifts, so a frame 40 on gives 64-71% and beyond ~24
    frames the overlap in *metric* terms can vanish entirely even though the
    scene is unchanged. A gate calibrated on adjacent frames (0.8) therefore
    rejects exactly the long-gap registrations this exists to make. 0.55 is
    below the lowest legitimate reading measured.

    The other population does not separate cleanly. A wrist camera riding a
    moving arm lands at 23-70%, which overlaps the fixed camera's long-gap
    range at the top; with the correspondence being the pixel grid there is no
    overlap left to fit, so the RANSAC returns a small, plausible-looking
    transform either way. Which camera is fixed is therefore something the
    *caller* has to know and declare, not something this function can infer --
    it can only refuse a frame that fits badly.

    Returns `None` — "do not align, keep the frame unaligned" — when the frame
    cannot be trusted: too few commonly valid pixels to sample or to fit an SVD,
    too few inliers, or a transform that is outright absurd (scale past
    `max_scale_error`, rotation past `max_rotation_deg`, or a translation past
    `max_translation_frac` of the anchor's median depth).

    Those three are **sanity rails, not discriminators**, and they are wide on
    purpose. The scale one especially: it is tempting to read a transform that
    should be near the identity as evidence the fit went wrong, but for a fixed
    camera the transform *is* the thing being corrected — the model's own drift
    — and that drift is large. Measured over three LIBERO episodes, a frame
    registered against its episode's anchor fits a scale of 0.97 to 1.46, a
    rotation up to 2.9 degrees and a translation up to 9% of the scene depth,
    all at 45-73% inliers and all correct. A rail at 0.1 rejects nearly every
    long-gap frame an episode has — which is what it did before it was widened.
    `min_inliers` is what distinguishes a real fit from an invented one; these
    only catch a fit that is not a rigid motion of a scene at all.

    Weights are `1 / depth` (see the module header), so the fit is dominated by
    the near, well-resolved pixels. Points are subsampled to at most `sample`
    with `rng` (default `default_rng(DEFAULT_SEED)`) to bound the cost of the
    RANSAC's 3x3 SVDs.

    `hypothetical_size` is the file's minimal-sample 10, *not* a large fraction
    of `sample`, and that is the difference between aligning and not. RANSAC
    needs each hypothesis to be drawn from inliers to fit anything exact; at
    the 70% inlier ratio a long-gap frame gives, a 2000-point hypothesis is 30%
    outliers and every one of them fits a biased answer, so the consensus it
    converges on is the wrong one (measured: 36% inliers where the minimal
    sample reads 70%). The iteration count is what buys the odds back — a clean
    10-draw is 3% likely at that ratio, so 200 of them is what finds one.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)

    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(anchor_points, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    d_anchor = np.asarray(anchor_depth, dtype=np.float64).reshape(-1)

    # A pixel only corresponds if both maps are usable there: a NaN or a
    # non-positive depth means the model had nothing to say, and it would poison
    # the fit rather than merely be ignored.
    common = (
        np.isfinite(p).all(axis=1)
        & np.isfinite(q).all(axis=1)
        & np.isfinite(d)
        & np.isfinite(d_anchor)
        & (d > 0)
        & (d_anchor > 0)
    )
    n_common = int(common.sum())
    if n_common < hypothetical_size or n_common < 3:
        return None

    idx = np.flatnonzero(common)
    if n_common > sample:
        idx = rng.choice(idx, size=sample, replace=False)
    p_s, q_s, d_s = p[idx], q[idx], d[idx]

    (scale, rotation, translation), inliers = rigid_registration_ransac(
        p_s,
        q_s,
        1.0 / d_s,
        max_iters=max_iters,
        hypothetical_size=hypothetical_size,
        inlier_thresh=inlier_thresh,
    )
    inlier_frac = float(np.mean(inliers))

    scale = float(scale)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)

    if inlier_frac < min_inliers:
        return None
    if abs(scale - 1.0) > max_scale_error:
        return None
    if _rotation_angle_deg(rotation) > max_rotation_deg:
        return None
    # The translation gate is relative to the anchor's depth scale, not absolute:
    # a similarity is only ever defined up to that scale, so a fixed metre
    # threshold would mean something different in every scene. The median is
    # taken over the commonly valid pixels so the gate tracks the part of the
    # anchor this registration actually saw.
    max_translation = max_translation_frac * float(np.median(d_anchor[common]))
    if float(np.linalg.norm(translation)) > max_translation:
        return None

    return Registration(
        scale=scale,
        rotation=rotation,
        translation=translation,
        inliers=inlier_frac,
        n_points=int(idx.shape[0]),
    )


def reproject_to_camera(
    points: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: np.ndarray,
    size: tuple[int, int],
    source_intrinsics: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply a similarity to a frame's geometry and re-render it into a camera.

    `points` (H, W, 3), `normals` (H, W, 3) and `valid` (H, W) describe the
    source frame, which was unprojected through `source_intrinsics` (its own
    frame's, unless it is the same camera as the target). Each point is mapped
    by ``p_world = scale * points @ rotation.T + translation`` — the convention
    `rigid_registration` returns — and re-rendered into the target camera given
    by `intrinsics`, a (3, 3) matrix in normalized uv units (``fx = k[0,0]``,
    ``fy = k[1,1]``, ``cx = k[0,2]``, ``cy = k[1,2]``), onto an output grid of
    `size` (H, W) with pixel centres at ``(j + 0.5) / W``, ``(i + 0.5) / H``.

    **The target grid is sampled backwards, not splatted forwards.** A forward
    splat is the obvious reading of "project every point and keep the nearest",
    and it is wrong here: a similarity is exactly invertible, but rounding each
    source point onto the nearest target pixel is not, so a rotation — which
    moves the source lattice off the target lattice — leaves a regular grid of
    pixels that nothing was rounded onto. On real frames that grid is plainly
    visible as a fine lattice over the whole image, and it is why this function
    does not do it. Sampling the target grid and inverting the map instead
    touches every target pixel exactly once and cannot leave a lattice.

    What the backward map needs is the depth of the surface the target ray hits,
    which is what it is solving for. It is found as a fixed point: a ray at an
    assumed depth inverts to a source pixel whose own depth, once transformed,
    gives the next assumption. The seed is the source's depth at the same pixel,
    which for the near-identity transforms this is used for is already close,
    and the iteration is what takes up the parallax the translation contributes.

    Occlusion is therefore not resolved by a z-buffer, and does not need to be
    here: with the transform near the identity — the only case this is used for,
    a fixed camera's own frame-to-frame drift — each target ray has one source
    pixel that is unambiguous except within sub-pixel of a depth discontinuity.
    A transform large enough to make that untrue would already have been refused
    by `register_to_anchor`'s gates.

    Normals are rotated by `rotation` alone: a similarity scales points but not
    directions, and normals are directions. (A non-uniform scale would need the
    inverse-transpose; a similarity does not.)

    Returns `(depth, normals, valid, hole_fraction)`. `depth` holds the target
    ray's depth in metres, 0.0 where nothing was found, and `normals` is 0.0
    there. Holes are not filled: `hole_fraction` is reported instead, so a bad
    registration stays visible to the caller rather than being papered over.
    """
    points = np.asarray(points)
    normals = np.asarray(normals)
    valid = np.asarray(valid)
    out_h, out_w = int(size[0]), int(size[1])
    src_h, src_w = valid.shape

    # Keep the source dtypes for the outputs: the identity round trip is
    # asserted with `np.array_equal`, and a stray float64 would also double the
    # memory of every rendered frame. The coordinate math below runs in float64
    # regardless, where the inverse of the map is exact.
    pdtype = points.dtype if np.issubdtype(points.dtype, np.floating) else np.float64
    ndtype = normals.dtype if np.issubdtype(normals.dtype, np.floating) else np.float64

    k_t = np.asarray(intrinsics, dtype=np.float64)
    k_s = k_t if source_intrinsics is None else np.asarray(source_intrinsics,
                                                           dtype=np.float64)
    fx_t, fy_t = float(k_t[0, 0]), float(k_t[1, 1])
    cx_t, cy_t = float(k_t[0, 2]), float(k_t[1, 2])
    fx_s, fy_s = float(k_s[0, 0]), float(k_s[1, 1])
    cx_s, cy_s = float(k_s[0, 2]), float(k_s[1, 2])

    flat_points = points.reshape(-1, 3).astype(np.float64, copy=False)
    flat_normals = normals.reshape(-1, 3)
    flat_valid = valid.reshape(-1).astype(bool)

    col_t, row_t = np.meshgrid(np.arange(out_w), np.arange(out_h))
    u = (col_t + 0.5) / out_w
    v = (row_t + 0.5) / out_h

    # Seed the fixed point with the source's own depth at the same pixel, then
    # let each pass correct it. Three passes is comfortably past convergence:
    # the error enters the inverse ray in proportion to depth, so the sequence
    # contracts by roughly that proportion each time.
    depth = np.where(valid, points[..., 2], 0.0).astype(np.float64)
    found = np.zeros(depth.shape, dtype=bool)
    source_of = np.zeros(depth.size, dtype=np.int64)
    for _ in range(3):
        ray = np.stack([(u - cx_t) / fx_t * depth,
                        (v - cy_t) / fy_t * depth,
                        depth], axis=-1)
        # Invert `p_world = scale * p @ rotation.T + translation`, which is
        # `p = ((p_world - translation) / scale) @ rotation` because
        # `(p @ R.T) @ R == p`.
        p_src = ((ray - np.asarray(translation, dtype=np.float64)) / scale) @ np.asarray(
            rotation, dtype=np.float64
        )
        z = p_src[..., 2]
        finite = np.isfinite(p_src).all(axis=-1)
        # Guard the divide rather than the result: a NaN would reach the cast
        # to an index below and is undefined there.
        z_safe = np.where(finite & (z > 0), z, 1.0)
        col_s = np.rint((fx_s * p_src[..., 0] / z_safe + cx_s) * src_w - 0.5)
        row_s = np.rint((fy_s * p_src[..., 1] / z_safe + cy_s) * src_h - 0.5)
        ok = (finite & (z > 0)
              & (col_s >= 0) & (col_s < src_w) & (row_s >= 0) & (row_s < src_h))
        # Clamped to a real pixel so the gather below is always in bounds; the
        # value is discarded wherever `ok` is false.
        idx = (np.where(ok, np.clip(row_s, 0, src_h - 1), 0.0).astype(np.int64) * src_w
               + np.where(ok, np.clip(col_s, 0, src_w - 1), 0.0).astype(np.int64))
        ok &= flat_valid[idx]
        # The target ray's depth, read off the source point this pass found.
        depth = np.where(ok, scale * flat_points[idx, 2]
                         + float(np.asarray(translation)[2]), 0.0)
        # Flat, so that the masks and source indices of the last pass can be
        # used directly as flat indices below.
        found, source_of = ok.reshape(-1), idx.reshape(-1)

    out_normals = np.zeros((out_h * out_w, 3), dtype=ndtype)
    out_normals[found] = (flat_normals[source_of[found]].astype(np.float64)
                          @ np.asarray(rotation, dtype=np.float64).T).astype(ndtype)
    hole_fraction = 1.0 - float(found.mean()) if found.size else 0.0
    return (depth.astype(pdtype), out_normals.reshape(out_h, out_w, 3),
            found.reshape(out_h, out_w), hole_fraction)
