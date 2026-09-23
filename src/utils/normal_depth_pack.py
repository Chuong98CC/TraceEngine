"""Pack a surface-normal map and a depth map into one uint8 RGB image.

Models that predict geometry give you two things per pixel: a surface normal
and a depth. Both fit in a single 3-channel PNG as ``[Oct_U, Oct_V, depth]``,
so an image model can consume them as an ordinary image rather than needing a
bespoke loader. :class:`NormalDepthPack` is the codec.

Both halves are lossy in ways that are chosen rather than incidental:

* A unit vector needs 2 channels but lives on a sphere, so the two normal
  channels must *fold*. The octahedral projection is used, which spreads its
  quantization error uniformly over the sphere. Encodings that instead assume a
  front-facing hemisphere (plain ``(nx, ny)``, and Lab/HSV variants of it)
  cannot represent the other hemisphere at all — measurably ~90 degrees wrong
  once a model's convention puts its data there.
* Depth is encoded in log space over a range, making the quantization relative
  rather than absolute, and the range adapts to the frame (see
  ``resolve_range``).

**The frame matters more than either.** The octahedral map is well conditioned
at its pole and degenerate at the opposite one — measured at 1.2 codes of
movement per degree of normal change in the best-conditioned band against 12.5
at the far pole. Every model parks its normals somewhere, and if that somewhere
is the far pole the output is *accurate but blocky*: a 0.56 degree step between
neighbouring pixels of one flat surface becomes a 359-code jump. So the caller
declares its model's ``pole`` — the direction it points surfaces that face the
camera — and the codec rotates that onto the projection pole. Use
:func:`estimate_pole` to measure a new model's convention once from a few
sample frames, hard-code it, and let :func:`pole_conditioning` confirm it.

The depth channel carries code 0 for invalid, 1..255 for data. Decoding needs
the range the image was encoded over, plus the model's depth scale; both live
in the sidecar written by :meth:`NormalDepthPack.scale_dict`, together with the
``pole``, so a packed image is self-describing.

That image is a data container, and spends all three channels on the two
signals. :func:`encode_dab` is the other trade: the same depth code ridden in
Lab's L channel, with a and b left to the *input frame's* own chroma. It
carries no normals and costs one code to the sRGB round trip, but the result
still reads as a picture, so a model trained on photographs can consume it.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: Default encoded depth range, in the model's own depth units. A rail rather
#: than a fixed range: ``resolve_range`` intersects these with the frame's own
#: range, so a scene inside them is encoded over its own tighter one.
DEFAULT_Z_MIN = 0.25
DEFAULT_Z_MAX = 3.0

#: Depth channel codes: 0 is reserved for invalid, valid data uses 1..255.
DEPTH_CODE_MIN = 1
DEPTH_CODE_MAX = 255
DEPTH_LEVELS = DEPTH_CODE_MAX - DEPTH_CODE_MIN  # 254 steps between those codes

#: MoGe's normal convention: it emits normals in the OpenCV camera frame, where
#: a surface facing the camera has nz = -1. Measured mean normal over six
#: cameras with unrelated viewpoints is (0.04, -0.06, -1.00) — a property of
#: the model, not of any one scene.
MOGE_POLE = np.array([0.0, 0.0, -1.0])

#: Normal change, in degrees, used to probe how many codes the encoding moves
#: per degree *where the data actually sits*. Small enough to be local, large
#: enough to clear the 8-bit floor.
CONDITIONING_PROBE_DEG = 2.0

#: Amplification above which a pole is reported as misplaced, in codes per
#: degree of normal change. Calibrated against measurement: real frames with
#: their pole declared correctly read 1.50-1.60, an ideal pole on a tight
#: cluster reads 2.10, the floor for any 2-channel encoding is 127.5*pi/180 =
#: 2.23, and the same real frames with the pole on the far side read
#: 4.70-16.24. Three is the geometric midpoint of that gap. The bias is toward
#: catching a misplaced pole rather than avoiding a false alarm: a wrong pole
#: costs no accuracy at all, so it stays invisible until it has wasted someone's
#: afternoon, whereas a spurious warning costs one line of output.
AMPLIFICATION_WARN = 3.0


def _rotation_to_pole(pole: np.ndarray) -> np.ndarray:
    """Minimal rotation carrying ``pole`` onto +z, the projection's pole.

    A proper rotation, so the transform never flips handedness. The
    antiparallel case is a one-parameter family (any of which conditions
    identically, since they differ by a turn about the projection axis); it is
    pinned to a 180-degree turn about x for reproducibility.
    """
    a = np.asarray(pole, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm == 0.0:
        raise ValueError("pole must be a non-zero vector")
    a = a / norm
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    s, c = float(np.linalg.norm(v)), float(a @ b)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / s**2)


def _rotate(normals: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a 3x3 rotation to a (..., 3) field, staying in float32.

    The cast is load-bearing: a float64 factor here promotes the whole
    projection to float64, where the ``+1e-8`` in the L1 norm below no longer
    rounds away and axis normals land one code short of the square's edge.
    """
    return np.einsum(
        "ij,...j->...i", rotation.astype(np.float32, copy=False), normals
    )


def _octahedral_uv(normals: np.ndarray) -> np.ndarray:
    """Project unit normals to octahedral coordinates in [-1, 1]^2."""
    l1_norm = np.sum(np.abs(normals), axis=-1, keepdims=True) + 1e-8
    p = normals[..., :2] / l1_norm

    # Unfold the hemisphere away from the pole onto the square's outer corners.
    # The sign must be +/-1, never np.sign: sign(0) is 0, which would zero the
    # coordinate the fold just derived from the other one.
    back = normals[..., 2] < 0
    p[back] = (1.0 - np.abs(p[back, ::-1])) * np.where(p[back] >= 0.0, 1.0, -1.0)
    return p


def _octahedral_normals(uv: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_octahedral_uv`: coordinates in [-1, 1]^2 -> unit normals."""
    u, v = uv[..., 0], uv[..., 1]
    z = 1.0 - (np.abs(u) + np.abs(v))

    below = z < 0.0
    sign_u = np.where(u >= 0.0, 1.0, -1.0)
    sign_v = np.where(v >= 0.0, 1.0, -1.0)

    x, y = u.copy(), v.copy()
    x[below] = (1.0 - np.abs(v[below])) * sign_u[below]
    y[below] = (1.0 - np.abs(u[below])) * sign_v[below]

    normals = np.stack([x, y, z], axis=-1)
    return normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)


def _depth_encode(
    depth_z: np.ndarray, z_min: float, z_max: float
) -> np.ndarray:
    """``depth_z`` -> uint8 codes 1..255. Code 0 is left reserved for invalid."""
    lo, hi = np.log(z_min), np.log(z_max)
    norm = np.clip((np.log(depth_z) - lo) / (hi - lo), 0.0, 1.0)
    # 1 + norm * 254 floors into 1..255; the cast truncates toward zero.
    return (DEPTH_CODE_MIN + norm * DEPTH_LEVELS).astype(np.uint8)


def _depth_decode(code: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """uint8 codes -> ``depth_z``. Code 0 is invalid and decodes to 0.0."""
    lo, hi = np.log(z_min), np.log(z_max)
    depth_z = lo + (code.astype(np.float32) - DEPTH_CODE_MIN) / DEPTH_LEVELS * (
        hi - lo
    )
    return np.where(code > 0, np.exp(depth_z), 0.0).astype(np.float32)


def _tangent(normals: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each normal, chosen deterministically.

    Picks whichever of the z and x axes is further from parallel, so the cross
    product never collapses.
    """
    reference = np.where(
        np.abs(normals[..., 2:3]) < 0.9,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    tangent = np.cross(normals, reference)
    return tangent / np.maximum(
        np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-8
    )


@dataclass(frozen=True)
class DepthScale:
    """Grounds a model's depth to metres: ``metric = (depth_z + shift) * metric_scale``.

    ``DepthScale()`` is the identity and covers a model that already predicts
    metres — which is why the codec has no metric-vs-relative mode flag. The
    affine pair covers a model whose depth is scale-free until something
    grounds it.
    """

    shift: float = 0.0
    metric_scale: float = 1.0

    def to_metric(self, depth_z: np.ndarray) -> np.ndarray:
        """The model's depth -> metres."""
        return (np.asarray(depth_z) + self.shift) * self.metric_scale

    def from_metric(self, metres: np.ndarray) -> np.ndarray:
        """Metres -> the model's depth. Inverse of :meth:`to_metric`."""
        return np.asarray(metres) / self.metric_scale - self.shift

    @classmethod
    def from_dict(cls, sidecar) -> "DepthScale":
        """Read a scale back out of a :meth:`NormalDepthPack.scale_dict`."""
        return cls(
            shift=float(sidecar["shift"]),
            metric_scale=float(sidecar["metric_scale"]),
        )


@dataclass(frozen=True)
class PoleConditioning:
    """How well a declared pole suits a model's actual normals.

    Attributes:
        centre_angle_deg: angle between the normals' mean direction and the
            pole. Near 0 is good.
        spread_p999_deg: p99.9 of the angular spread about the pole.
        amplification: codes of movement per degree of normal change, measured
            where the data sits. This is the number that matters — a pole on
            the far side leaves accuracy untouched but multiplies this, which
            is what turns a flat surface into hard-edged colour blocks.
        ok: ``amplification`` within :data:`AMPLIFICATION_WARN`.
    """

    centre_angle_deg: float
    spread_p999_deg: float
    amplification: float
    ok: bool


def estimate_pole(
    normals: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    """Estimate a model's pole from sample normals: their mean direction.

    Run a few frames of a new model through this, then hard-code the answer as
    the ``pole`` argument. Fixing it (rather than re-deriving it per image)
    keeps the encoding frame stable, so identical codes mean identical normals
    from frame to frame.

    Args:
        normals: (..., 3) normals from the model, any shape.
        valid: matching bool mask, or None to use every finite normal.

    Returns:
        (3,) unit vector, float64.
    """
    normals = np.asarray(normals, dtype=np.float32)
    usable = np.all(np.isfinite(normals), axis=-1)
    if valid is not None:
        usable &= np.asarray(valid, dtype=bool)

    selected = normals[usable]
    if selected.size == 0:
        raise ValueError("no usable normals to estimate a pole from")
    selected = selected / np.maximum(
        np.linalg.norm(selected, axis=-1, keepdims=True), 1e-9
    )

    mean = selected.mean(axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float64)


def pole_conditioning(
    normals: np.ndarray,
    pole: np.ndarray = MOGE_POLE,
    valid: np.ndarray | None = None,
) -> PoleConditioning:
    """Report how well ``pole`` suits ``normals``, in the model's own frame.

    Measures the real encoding: the normals are tilted by
    :data:`CONDITIONING_PROBE_DEG` and the resulting code movement is read off
    the octahedral projection, so the number is the amplification this data
    would actually get rather than a proxy for it.
    """
    normals = np.asarray(normals, dtype=np.float32)
    usable = np.all(np.isfinite(normals), axis=-1)
    if valid is not None:
        usable &= np.asarray(valid, dtype=bool)

    selected = normals[usable]
    if selected.size == 0:
        raise ValueError("no usable normals to report on")
    selected = selected / np.maximum(
        np.linalg.norm(selected, axis=-1, keepdims=True), 1e-9
    )

    pole = np.asarray(pole, dtype=np.float64)
    pole = pole / np.linalg.norm(pole)

    spread = np.degrees(np.arccos(np.clip(selected @ pole, -1.0, 1.0)))
    mean = selected.mean(axis=0)
    centre = float(
        np.degrees(
            np.arccos(np.clip(mean @ pole / np.linalg.norm(mean), -1.0, 1.0))
        )
    )

    rotation = _rotation_to_pole(pole)
    step = np.radians(CONDITIONING_PROBE_DEG)
    tilted = (np.cos(step) * selected + np.sin(step) * _tangent(selected)).astype(
        np.float32
    )
    before = _octahedral_uv(_rotate(selected, rotation))
    after = _octahedral_uv(_rotate(tilted, rotation))
    # uv spans [-1, 1], so *127.5 puts the movement in code units.
    amplification = float(
        np.linalg.norm((after - before) * 127.5, axis=-1).mean()
        / CONDITIONING_PROBE_DEG
    )

    return PoleConditioning(
        centre_angle_deg=centre,
        spread_p999_deg=float(np.percentile(spread, 99.9)),
        amplification=amplification,
        ok=amplification < AMPLIFICATION_WARN,
    )


class NormalDepthPack:
    """Codec for a packed ``[Oct_U, Oct_V, depth]`` image.

    Args:
        pole: the direction this model points surfaces that face the camera,
            in the model's own frame. See :func:`estimate_pole`.
        z_min: lower rail for the encoded depth range, model depth units.
        z_max: upper rail for the encoded depth range, model depth units.
        adaptive: whether to intersect ``[z_min, z_max]`` with each frame's own
            depth range (see :meth:`resolve_range`). Turn off for a fixed,
            frame-independent encoding.
        check_conditioning: whether :meth:`encode` checks the declared pole
            against the data and warns when it is misplaced. Costs one extra
            pass over the normals.
    """

    def __init__(
        self,
        pole: np.ndarray = MOGE_POLE,
        z_min: float = DEFAULT_Z_MIN,
        z_max: float = DEFAULT_Z_MAX,
        adaptive: bool = True,
        check_conditioning: bool = True,
    ) -> None:
        if not 0.0 < z_min < z_max:
            raise ValueError(f"need 0 < z_min < z_max, got {z_min}, {z_max}")
        self.pole = np.asarray(pole, dtype=np.float64)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.adaptive = bool(adaptive)
        self.check_conditioning = bool(check_conditioning)
        self._rotation = _rotation_to_pole(self.pole)

    # ------------------------------------------------------------------ range

    def resolve_range(
        self, depth_z: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[float, float]:
        """Tighten ``[z_min, z_max]`` to the range this frame actually uses.

        Quantization error is set by the encoded span, so a scene occupying a
        narrow slice of the rails should be encoded over that slice — measured
        over the test frames, at 0.29-0.62% against 0.97% for a fixed span.

        The rails stay hard: the result never escapes them. ``(z_min, z_max)``
        come back unchanged when the data cannot tighten them, which is the
        no-valid-pixels case and the disjoint case (intersecting there would
        invert the range). Both fall back to the rails, which clip rather than
        fail.
        """
        if not self.adaptive:
            return self.z_min, self.z_max

        depth_z = np.asarray(depth_z)
        usable = np.isfinite(depth_z) & (depth_z > 0.0)
        if valid is not None:
            usable &= np.asarray(valid, dtype=bool)
        if not usable.any():
            return self.z_min, self.z_max

        lo = max(self.z_min, float(depth_z[usable].min()))
        hi = min(self.z_max, float(depth_z[usable].max()))
        if not lo < hi:
            # Data clear of the rails, or a scene flat enough that the span
            # would divide by zero.
            return self.z_min, self.z_max
        return lo, hi

    # -------------------------------------------------------------- validity

    @staticmethod
    def valid_mask(
        depth_z: np.ndarray, valid: np.ndarray | None = None
    ) -> np.ndarray:
        """Pixels the depth channel can represent.

        Non-finite and non-positive depths are excluded whether or not a mask
        is handed in: ``log`` of either is undefined, and left alone they would
        clip to the rail instead of preserving the invalid 0.
        """
        depth_z = np.asarray(depth_z)
        mask = np.isfinite(depth_z) & (depth_z > 0.0)
        if valid is not None:
            mask &= np.asarray(valid, dtype=bool)
        return mask

    # ----------------------------------------------------------------- codec

    def _warn_if_pole_misplaced(
        self, normals: np.ndarray, mask: np.ndarray
    ) -> None:
        """Warn when the declared pole leaves the data badly conditioned.

        Worth warning about because it is invisible otherwise: a misplaced pole
        does not hurt accuracy at all, so the decoded normals look fine. It
        only shows up as blocky output.
        """
        report = pole_conditioning(normals, self.pole, mask)
        if report.ok:
            return
        warnings.warn(
            f"normals sit {report.centre_angle_deg:.0f} deg from the declared "
            f"pole and need {report.amplification:.1f} codes of movement per "
            f"degree of normal change (warn above {AMPLIFICATION_WARN:.0f}) — "
            f"the image will be blocky even though it decodes accurately. "
            f"Check that pole={np.round(self.pole, 4).tolist()} matches this "
            f"model's convention; see estimate_pole.",
            stacklevel=3,
        )

    def encode(
        self,
        normals: np.ndarray,
        depth_z: np.ndarray,
        valid: np.ndarray | None = None,
        z_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Pack normals + depth into a (H, W, 3) uint8 image.

        Args:
            normals: (H, W, 3) unit normals, in the model's own frame.
            depth_z: (H, W) the model's depth, in its own units. Metric models
                pass metres; affine models pass the scale-free depth that
                ``DepthScale`` grounds.
            valid: (H, W) bool, or None to take every representable depth.
            z_range: the ``(z_min, z_max)`` to encode over. Resolved from the
                data when omitted; pass :meth:`resolve_range`'s result to
                record exactly the range that was used.

        Returns:
            (H, W, 3) uint8 image. Pixels that are not valid pack to (0, 0, 0).
        """
        normals = np.asarray(normals, dtype=np.float32)
        depth_z = np.asarray(depth_z, dtype=np.float32)
        if normals.shape[:2] != depth_z.shape or normals.shape[-1] != 3:
            raise ValueError(
                f"expected normals (H, W, 3) and depth_z (H, W), got "
                f"{normals.shape} and {depth_z.shape}"
            )

        mask = self.valid_mask(depth_z, valid)
        if z_range is None:
            z_range = self.resolve_range(depth_z, mask)
        z_min, z_max = z_range

        if self.check_conditioning:
            self._warn_if_pole_misplaced(normals, mask)

        uv = _octahedral_uv(_rotate(normals, self._rotation))
        u = ((uv[..., 0] + 1.0) * 0.5 * 255.0).astype(np.uint8)
        v = ((uv[..., 1] + 1.0) * 0.5 * 255.0).astype(np.uint8)
        # The placeholder keeps a non-finite depth away from the log; those
        # pixels are zeroed wholesale below regardless.
        d = _depth_encode(np.where(mask, depth_z, 1.0).astype(np.float64), z_min, z_max)

        packed = np.stack([u, v, d], axis=-1)
        packed[~mask] = 0
        return packed

    def decode(
        self,
        image: np.ndarray,
        z_min: float,
        z_max: float,
        pole: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpack a (H, W, 3) uint8 image back to normals + depth.

        Args:
            image: (H, W, 3) uint8, ``[Oct_U, Oct_V, depth]``.
            z_min: lower bound the image was encoded over. Must match the
                encode side, or the depths come back on the wrong scale.
            z_max: upper bound the image was encoded over.
            pole: the pole the image was encoded with. Defaults to this
                instance's; pass the sidecar's to read a foreign image.

        Returns:
            ``(normals (H, W, 3) float32, depth_z (H, W) float32, valid
            (H, W) bool)``. ``valid`` is the depth channel being non-zero —
            test it, not the normals: an invalid pixel's depth is 0.0, which is
            a real depth, and its normals decode to a plausible-looking
            direction.
        """
        image = np.asarray(image, dtype=np.uint8)
        rotation = self._rotation if pole is None else _rotation_to_pole(pole)

        uv = image[..., :2].astype(np.float32) / 255.0 * 2.0 - 1.0
        normals = _rotate(_octahedral_normals(uv), rotation.T)

        code = image[..., 2]
        return (
            normals.astype(np.float32),
            _depth_decode(code, z_min, z_max),
            code > 0,
        )

    # --------------------------------------------------------------- sidecar

    def scale_dict(
        self,
        scale: DepthScale,
        z_min: float,
        z_max: float,
        shape: tuple[int, int],
        **extra: object,
    ) -> dict:
        """Build the sidecar that makes a packed image decodable, for ``savez``.

        Args:
            scale: the :class:`DepthScale` that grounds this model's depth.
            z_min: lower bound the image was encoded over.
            z_max: upper bound the image was encoded over.
            shape: ``(H, W)`` of the packed image.
            **extra: anything else worth keeping beside the image — intrinsics,
                a frame index. Stored verbatim, so a caller can put back
                whatever the codec does not own.

        Returns:
            A dict suitable for ``np.savez``.
        """
        height, width = shape
        sidecar = {
            "shift": np.float64(scale.shift),
            "metric_scale": np.float64(scale.metric_scale),
            "z_min": np.float64(z_min),
            "z_max": np.float64(z_max),
            "image_h": np.int32(height),
            "image_w": np.int32(width),
            # Recorded rather than inferred: without it a stale image decodes
            # silently wrong, and a reader would have to be configured to match
            # the writer.
            "pole": np.asarray(self.pole, dtype=np.float64),
        }
        sidecar.update(extra)
        return sidecar

    @staticmethod
    def load_scale(path: str | Path) -> dict:
        """Load a sidecar written by :meth:`scale_dict` into a plain dict."""
        with np.load(path) as handle:
            return {name: handle[name] for name in handle.files}

    def decode_with_scale(
        self, image: np.ndarray, sidecar
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode a packed image using its sidecar, pole included.

        Honours the pole the sidecar records rather than this instance's, so a
        reader does not have to be configured to match the writer.
        """
        pole = sidecar["pole"] if "pole" in sidecar else self.pole
        return self.decode(
            image, float(sidecar["z_min"]), float(sidecar["z_max"]), pole=pole
        )


# --------------------------------------------------------- the Dab carrier
# The packed image above is a data container: it spends all three channels on
# the two signals and so does not look like a photograph.  The Dab carrier is
# the other trade -- it carries only the depth, riding in Lab's L channel, and
# leaves a and b to the input frame's own chroma.  The result still reads as a
# picture (a monochrome one, tinted by the original), which is what makes it
# usable by an image model that was trained on photographs.
#
# The codec does not change, only the carrier: the depth channel is the same
# code over the same range, so a `_scale.npz` sidecar decodes either image and
# a caller can write both for one frame.
#
# What the carrier costs is the sRGB round trip -- L is written as a colour and
# read back out of one.  (L, a, b) is not always a representable colour: a
# saturated pixel's chroma admits only a narrow band of L, and forcing L to the
# depth code leaves the gamut, where the clip moves L back.  Measured on real
# frames, letting that happen loses up to 12 codes (7.7% of the depth) on the
# most saturated frame, against 1 code everywhere once the chroma is reduced to
# fit.  So the encoder bisects the largest chroma scale that keeps the target L
# representable, and only the pixels that need it are washed out.

#: The depth code, rescaled onto Lab's 0-100 luminance axis. Kept for the
#: gamut tolerance, which is naturally expressed in codes.
DAB_L_PER_CODE = 100.0 / DEPTH_CODE_MAX

#: The luminance at each end of the depth ramp, named for the end and not for
#: which one is bright -- the two are not in ascending order here, and a caller
#: can swap them to run the ramp the other way.
#:
#: The span is held inside 0-100 on purpose: at the full range the nearest
#: surface lands near-black and the farthest near-white, so the image reads as
#: an exposure problem rather than as a depth map. It also costs depth
#: accuracy, because a narrower span makes each code a smaller step in L for
#: the 8-bit round trip to resolve -- measured over the ramp, 0-100 gives a max
#: error of 0.98%, 10-90 the same 0.98%, 20-85 1.97%, and 40-65 3.93%. The
#: defaults below sit at 2 and 95, which is the wide end of that range and so
#: keeps the accuracy; they are a rail, not a target.
DAB_L_FAR = 95.0
DAB_L_NEAR = 2.0

#: Bisection steps used to find a pixel's chroma scale.  Ten halves the chroma
#: to 0.1%, well below the uint8 rounding that follows.
DAB_GAMUT_STEPS = 10

#: How many codes of round-trip error a pixel may show before its chroma is
#: reduced.  One, not zero: the sRGB transfer curve is so steep near black that
#: a single uint8 step there is worth several codes, so insisting on an exact
#: code would spend the whole frame's chroma chasing a quantization floor the
#: format has anyway.
DAB_CODE_TOLERANCE = 1.0

#: How much of the frame's high-frequency L to add back on top of the depth.
#: Zero would leave a bare depth ramp, which reads as a depth map rather than
#: as a picture; this much restores the surface markings.
#:
#: It is not free. The markings and the depth payload share one channel, so
#: whatever is added here *is* the depth error: measured on a real frame, 0.1
#: takes the carrier's guarantee from 1.0% to a p99 of 2.9% and a max of 7.9%.
#: Raising it raises that in proportion, so the exactness of the depth and the
#: richness of the image are the same dial.
DAB_DETAIL_ALPHA = 0.1

#: The high-pass behind that term: ``L - bilateral(L)``, edge preserving, so
#: strong edges cancel out of the residual instead of being re-added. That is
#: what keeps its amplitude -- and the accuracy it costs -- far below a
#: Gaussian high-pass of the same visual strength.
DAB_DETAIL_D = 15
DAB_DETAIL_SIGMA_COLOR = 0.2
DAB_DETAIL_SIGMA_SPACE = 9


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Float RGB in [0, 1] -> cv2 Lab (L 0-100, a/b signed)."""
    return cv2.cvtColor(np.asarray(rgb, dtype=np.float32), cv2.COLOR_RGB2LAB)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """cv2 Lab -> float RGB.

    Note that cv2 *clamps* this to [0, 1]: an out-of-gamut colour comes back
    with its negative channels pinned to exactly 0, not left negative.  So a
    gamut test cannot be written as "is any channel negative" -- it never fires.
    That is why :func:`_dab_code_of` checks the round trip's actual payload
    instead of trying to detect the clip.
    """
    return cv2.cvtColor(np.asarray(lab, dtype=np.float32), cv2.COLOR_Lab2RGB)


def _dab_blend(target_l: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A Lab triple -> the uint8 RGB it becomes, clip and rounding included.

    Deliberately runs the *whole* encode path, quantization and all: the near
    black end of the range is where one uint8 step is worth several codes, so a
    float-only check would call a pixel representable that does not survive.

    The clamp is dead weight today -- cv2 already pins the result to [0, 1], as
    :func:`_lab_to_rgb` notes -- but it is what keeps the cast below honest. A
    value above 1.0 reaching a uint8 cast wraps rather than saturating, turning
    a white pixel into a nearly black one, so the guard stays even though it
    never fires.
    """
    rgb = np.clip(_lab_to_rgb(np.stack([target_l, a, b], axis=-1)), 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def _dab_l_of(blended: np.ndarray) -> np.ndarray:
    """The L a blended image reads back as, through its own round trip."""
    return _rgb_to_lab(blended.astype(np.float32) / 255.0)[..., 0]


def _dab_detail(
    l_channel: np.ndarray,
    d: int = DAB_DETAIL_D,
    sigma_color: float = DAB_DETAIL_SIGMA_COLOR,
    sigma_space: float = DAB_DETAIL_SIGMA_SPACE,
) -> np.ndarray:
    """The high-frequency part of L, in L* units.

    ``L - bilateral(L)``: surface markings stay, slow lighting goes. A
    bilateral rather than a Gaussian because it preserves edges -- a strong
    edge cancels out of the residual instead of being readded, which is what
    makes this term cheap enough to spend depth accuracy on.
    """
    l_unit = l_channel / 100.0
    base = cv2.bilateralFilter(
        l_unit, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )
    return (l_unit - base) * 100.0


def _gamut_scale(
    target_l: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Per pixel, the largest chroma scale in [0, 1] whose L survives.

    The predicate is the payload itself -- encode, decode, and ask whether the L
    came back within :data:`DAB_CODE_TOLERANCE` codes -- rather than a gamut
    proxy.  A proxy has to model cv2's clipping, and getting that wrong is
    silent: it leaves a saturated pixel encoding to a colour whose L is nowhere
    near the depth, measured at 37 codes out on a near-black blue.

    The test is on L rather than on the code because L is also where the
    optional detail layer lands; asking for the code back would have the
    reduction fight the detail it was just asked to add.

    Chroma is scaled rather than clamped per channel, so the hue survives and
    only the saturation is spent.  Well founded because reducing the scale moves
    the colour toward neutral grey, which round trips for every L, so ``lo`` is
    a verified-good answer even when nothing else is.
    """
    tolerance = DAB_CODE_TOLERANCE * DAB_L_PER_CODE
    lo = np.zeros_like(target_l)
    hi = np.ones_like(target_l)
    for _ in range(DAB_GAMUT_STEPS):
        mid = 0.5 * (lo + hi)
        back = _dab_l_of(_dab_blend(target_l, a * mid, b * mid))
        ok = np.abs(back - target_l) <= tolerance
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


def encode_dab(
    rgb: np.ndarray,
    depth_z: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    z_range: tuple[float, float],
    detail_alpha: float = DAB_DETAIL_ALPHA,
    detail_d: int = DAB_DETAIL_D,
    detail_sigma_color: float = DAB_DETAIL_SIGMA_COLOR,
    detail_sigma_space: float = DAB_DETAIL_SIGMA_SPACE,
    l_far: float = DAB_L_FAR,
    l_near: float = DAB_L_NEAR,
) -> np.ndarray:
    """Blend a depth map into a frame as ``Lab(depth + detail, a, b)``.

    Args:
        rgb: (H, W, 3) uint8 RGB, the frame whose chroma supplies a and b.
        depth_z: (H, W) the model's depth, in its own units — the same value
            :meth:`NormalDepthPack.encode` takes.
        valid: (H, W) bool, or None to take every representable depth.
        z_range: the ``(z_min, z_max)`` to encode over. Keyword-only and
            required, because it has to match the decode side exactly and a
            silent default would encode and decode on different scales.
        detail_alpha: how much of the frame's high-frequency L to add back over
            the depth, as a multiple of ``L - bilateral(L)``. Zero leaves a bare
            depth ramp; see :data:`DAB_DETAIL_ALPHA` for what the default costs
            in depth accuracy.
        detail_d, detail_sigma_color, detail_sigma_space: the bilateral the
            detail is taken against. ``sigma_color`` sets how much a strong
            edge is allowed into the residual, and so is the main dial on what
            the detail costs the depth.
        l_far, l_near: the L* the ramp runs between, nearest surface brightest.
            A property of the encoding, not of the codec, so a reader has to be
            told them -- see :func:`decode_dab`.

    Returns:
        (H, W, 3) uint8 RGB. Pixels that are not valid come back (0, 0, 0),
        the same sentinel the packed image uses.

    Requires no ``pole``: nothing here depends on the model's normal
    convention, and the normals are not carried at all.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth_z = np.asarray(depth_z, dtype=np.float32)
    if rgb.shape[:2] != depth_z.shape or rgb.shape[-1] != 3:
        raise ValueError(
            f"expected rgb (H, W, 3) and depth_z (H, W), got {rgb.shape} and "
            f"{depth_z.shape}"
        )
    z_min, z_max = z_range
    if not 0.0 < z_min < z_max:
        raise ValueError(f"need 0 < z_min < z_max, got {z_min}, {z_max}")
    # Either end may be the bright one, so the two are not ordered -- only
    # bounded and distinct. Above 0 keeps a valid pixel off pure black, which is
    # the invalid sentinel, so the two can never collide.
    if not (0.0 < l_far <= 100.0 and 0.0 < l_near <= 100.0 and l_far != l_near):
        raise ValueError(
            f"need two distinct values in (0, 100], got l_far={l_far}, "
            f"l_near={l_near}"
        )

    mask = NormalDepthPack.valid_mask(depth_z, valid)
    # The placeholder keeps a non-finite depth out of the log; those pixels are
    # zeroed wholesale below regardless.
    code = _depth_encode(
        np.where(mask, depth_z, 1.0).astype(np.float64), z_min, z_max
    ).astype(np.float32)
    # Code 1 is the nearest surface, and the nearest surface is the bright end.
    target_l = l_far + (l_near - l_far) * (
        (DEPTH_CODE_MAX - code) / (DEPTH_CODE_MAX - DEPTH_CODE_MIN)
    )

    lab = _rgb_to_lab(rgb.astype(np.float32) / 255.0)
    a, b = lab[..., 1], lab[..., 2]
    if detail_alpha:
        # Clamped to the ramp's ends, which is to say to the codes 1..255 that
        # mean "valid". Without it the detail can push a pixel at either end
        # past the ramp, and the code it then clamps to is not the one the
        # depth asked for -- the near rail comes back as a wrong distance and
        # the far end as "invalid". Only the extremes are affected, and there
        # the pixel is already on a bound, so suppressing its detail costs
        # nothing.
        # Sorted, because the ramp's ends are not in ascending order once the
        # far end is the bright one, and clip() given a min above its max
        # returns the max for every pixel -- quietly flattening the image to
        # one value instead of clamping it.
        ramp_low, ramp_high = sorted((l_far, l_near))
        target_l = np.clip(
            target_l + detail_alpha * _dab_detail(
                lab[..., 0], detail_d, detail_sigma_color, detail_sigma_space
            ),
            ramp_low,
            ramp_high,
        )
    scale = _gamut_scale(target_l, a, b)

    out = _dab_blend(target_l, a * scale, b * scale)
    out[~mask] = 0
    return out


def decode_dab(
    image: np.ndarray,
    z_min: float,
    z_max: float,
    l_far: float = DAB_L_FAR,
    l_near: float = DAB_L_NEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the depth a :func:`encode_dab` image carries.

    Args:
        image: (H, W, 3) uint8 RGB from :func:`encode_dab`.
        z_min: lower bound the image was encoded over. Must match the encode
            side, or the depths come back on the wrong scale.
        z_max: upper bound the image was encoded over.
        l_far: the L* the far end of the ramp was written at.
        l_near: the L* the near end was written at. Both must match the encode
            side for the same reason as ``z_min``/``z_max``; a reader should
            take them from the sidecar rather than guess.

    Returns:
        ``(depth_z (H, W) float32, valid (H, W) bool)``. Valid means the pixel
        is not the all-zero sentinel; the depth of an invalid pixel is 0.0.
    """
    image = np.asarray(image, dtype=np.uint8)
    l_channel = _rgb_to_lab(image.astype(np.float32) / 255.0)[..., 0]

    # The encode ramp, read backwards: L -> how far along the ramp -> code.
    along = (l_channel - l_far) / (l_near - l_far)
    # Clamped to 1..255, never 0: a pixel at or past the near end of the ramp
    # would otherwise clamp to code 0, which _depth_decode reads as "no depth"
    # -- a valid pixel coming back as 0.0 while the sentinel test calls it
    # valid, and the two disagreeing about the same pixel.
    code = np.clip(
        np.round(DEPTH_CODE_MAX - along * (DEPTH_CODE_MAX - DEPTH_CODE_MIN)),
        DEPTH_CODE_MIN,
        DEPTH_CODE_MAX,
    ).astype(np.uint8)
    # Not ``code > 0`` any more: a sentinel pixel's L is *below* the ramp, so it
    # clamps to the far end rather than to a zero code, and would come back as a
    # perfectly plausible distance. The sentinel is the all-zero pixel itself.
    return _depth_decode(code, z_min, z_max), np.any(image != 0, axis=-1)


# ------------------------------------------------------- the Alb_normal carrier
# R carries an albedo map (stored, not encoded), G and B carry the normal's nx
# and ny.  There is no depth in this one at all.
#
# Two channels is one short of a sphere, so the hemisphere is not stored: the
# reader takes the normal to be on the declared pole's side.  That is lossy for
# the pixels that really do point the other way -- measured over six real
# frames, 0.184% land on the far side at more than 10 degrees out and the worst
# reaches 49.  The trade is deliberate: the common case is exact, and the
# octahedral carrier is still there when the whole sphere has to survive.
#
# Validity rides on the (nx, ny) pair rather than on a reserved code, because a
# unit normal's nx and ny cannot both be -1: that would need |n| = sqrt(2).  So
# both channels being zero is unreachable for real data, and the albedo channel
# keeps all 256 of its levels instead of giving one up to a sentinel.


def encode_alb_normal(
    albedo: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Blend an albedo map with a normal's x and y into one uint8 RGB.

    Args:
        albedo: (H, W) uint8, carried verbatim in the R channel.
        normals: (H, W, 3) unit normals. Only nx and ny are carried.
        valid: (H, W) bool, or None to take every finite normal.

    Returns:
        (H, W, 3) uint8 RGB: ``[albedo, nx, ny]``. Pixels that are not valid
        come back (0, 0, 0).

    Takes no ``pole``: nothing is rotated here, the components are stored as
    the model gives them. The pole is the *reader's* business -- see
    :func:`decode_alb_normal`.
    """
    albedo = np.asarray(albedo, dtype=np.uint8)
    normals = np.asarray(normals, dtype=np.float32)
    if normals.shape[:2] != albedo.shape or normals.shape[-1] != 3:
        raise ValueError(
            f"expected albedo (H, W) and normals (H, W, 3), got "
            f"{albedo.shape} and {normals.shape}"
        )

    mask = np.isfinite(normals).all(axis=-1)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)

    xy = np.clip(normals[..., :2], -1.0, 1.0)
    image = np.stack(
        [
            albedo,
            ((xy[..., 0] + 1.0) * 0.5 * 255.0).round().astype(np.uint8),
            ((xy[..., 1] + 1.0) * 0.5 * 255.0).round().astype(np.uint8),
        ],
        axis=-1,
    )
    image[~mask] = 0
    return image


def decode_alb_normal(
    image: np.ndarray, pole: np.ndarray = MOGE_POLE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the albedo and normals an :func:`encode_alb_normal` image holds.

    Args:
        image: (H, W, 3) uint8 from :func:`encode_alb_normal`.
        pole: the direction this model points camera-facing surfaces. Supplies
            the hemisphere the missing nz sign is on.

    Returns:
        ``(normals (H, W, 3) float32, albedo (H, W) uint8, valid (H, W) bool)``.
        The albedo is exact. The normals are exact for the pole's hemisphere
        and mirrored for the far one, which is not recoverable from the image.
    """
    image = np.asarray(image, dtype=np.uint8)
    pole = np.asarray(pole, dtype=np.float64)

    g, b = image[..., 1], image[..., 2]
    # Both zero is the sentinel: nx and ny cannot both be -1 for a unit normal.
    # It is also what the z sign below has to key off, so a pole with no z at
    # all leaves the hemisphere undecidable rather than merely awkward.
    if pole[2] == 0.0:
        raise ValueError(
            f"pole {pole.tolist()} lies in the xy-plane, so it cannot say "
            f"which hemisphere the normals are in: this carrier stores nx and "
            f"ny only and resolves the missing z from the pole"
        )

    nx = g.astype(np.float32) / 255.0 * 2.0 - 1.0
    ny = b.astype(np.float32) / 255.0 * 2.0 - 1.0
    nz = np.sqrt(np.maximum(1.0 - nx * nx - ny * ny, 0.0))
    normals = np.stack([nx, ny, nz * np.sign(pole[2])], axis=-1)
    # Rounding can push nx*nx + ny*ny just past 1, which flattens nz to 0 and
    # leaves the vector short of unit length.
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)

    valid = (g != 0) | (b != 0)
    return normals.astype(np.float32), np.where(valid, image[..., 0], 0), valid


# -------------------------------------------------------- the L_normal carrier
# The frame's own luminance stays in L; the normal's nx and ny go into a and b.
# Nothing else is carried -- no depth, no albedo.
#
# Of the three normal-carrying channels this is the worst place to put a
# payload. a and b are signed chroma, and how much of them sRGB can represent
# depends on L: measured, +-80 and +-56 at L* 50 but +-8 at L* 95 and +-8 at
# L* 5. So the capacity is set by each pixel's own brightness, and the pair
# survives only where the frame happens to be mid-grey. On a real frame that is
# about a third of it; the rest comes back 10-25 degrees out. --save-Alb-norm
# carries the same normals in G and B, which span the full byte range, and
# lands at 0.22 degrees. Use this one for what the frame looks like, not when
# the normals have to survive.

#: How far nx and ny are scaled into a and b. 56 is the largest that fits the
#: gamut's narrower channel at mid luminance; past it the mid-tones clip too,
#: and the whole frame loses the pair rather than just its bright and dark ends.
LN_SCALE = 56.0


def encode_l_normal(
    rgb: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    scale: float = LN_SCALE,
) -> np.ndarray:
    """Put a normal's nx and ny into a frame's Lab chroma, keeping its L.

    Args:
        rgb: (H, W, 3) uint8 RGB, the frame supplying the luminance.
        normals: (H, W, 3) unit normals. Only nx and ny are carried.
        valid: (H, W) bool, or None to take every finite normal.
        scale: how far nx and ny are stretched into a and b. Must match the
            reader's, and cannot exceed 127 since that is the range's end.

    Returns:
        (H, W, 3) uint8 RGB. Pixels that are not valid come back (0, 0, 0).

    The gamut clips the chroma per pixel, and there is nothing to trade against
    it here -- a and b *are* the payload. A pixel whose normals do not fit its
    own luminance comes back at the nearest pair that does.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    normals = np.asarray(normals, dtype=np.float32)
    if normals.shape[:2] != rgb.shape[:2] or normals.shape[-1] != 3:
        raise ValueError(
            f"expected rgb (H, W, 3) and normals (H, W, 3), got {rgb.shape} "
            f"and {normals.shape}"
        )
    if not 0.0 < scale <= 127.0:
        raise ValueError(f"need 0 < scale <= 127, got {scale}")

    mask = np.isfinite(normals).all(axis=-1)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)

    lab = _rgb_to_lab(rgb.astype(np.float32) / 255.0)
    xy = np.clip(normals[..., :2], -1.0, 1.0) * scale
    blended = np.clip(
        _lab_to_rgb(np.stack([lab[..., 0], xy[..., 0], xy[..., 1]], axis=-1)),
        0.0,
        1.0,
    )
    out = (blended * 255.0).round().astype(np.uint8)
    out[~mask] = 0
    return out


def decode_l_normal(
    image: np.ndarray,
    pole: np.ndarray = MOGE_POLE,
    *,
    scale: float = LN_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the normals an :func:`encode_l_normal` image carries.

    Args:
        image: (H, W, 3) uint8 RGB from :func:`encode_l_normal`.
        pole: the direction this model points camera-facing surfaces. Supplies
            the hemisphere the missing nz sign is on.
        scale: the scale the writer used. Must match, or the pair comes back
            at the wrong magnitude -- and a wrong magnitude is a wrong normal,
            not a wrong brightness, so nothing about it looks off.

    Returns:
        ``(normals (H, W, 3) float32, valid (H, W) bool)``. Valid means the
        pixel is not the all-zero sentinel. The luminance is not returned: it
        is the frame's own, and carries nothing the image does not.
    """
    image = np.asarray(image, dtype=np.uint8)
    pole = np.asarray(pole, dtype=np.float64)
    if pole[2] == 0.0:
        raise ValueError(
            f"pole {pole.tolist()} lies in the xy-plane, so it cannot say "
            f"which hemisphere the normals are in: this carrier stores nx and "
            f"ny only and resolves the missing z from the pole"
        )

    lab = _rgb_to_lab(image.astype(np.float32) / 255.0)
    nx, ny = lab[..., 1] / scale, lab[..., 2] / scale
    nz = np.sqrt(np.maximum(1.0 - nx * nx - ny * ny, 0.0)) * np.sign(pole[2])
    normals = np.stack([nx, ny, nz], axis=-1)
    # The gamut clips the pair inward, not outward, so this is only reached
    # when rounding pushes it a hair past the unit circle.
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)

    return normals.astype(np.float32), np.any(image != 0, axis=-1)


# ---------------------------------------------------------------- fused
# The packed image with an L* detail term laid over the depth channel:
# [Oct_U, Oct_V, d + alpha * R_L*].  The layout is deliberately the packed
# one, so the detail is the *only* difference -- which means decode() reads a
# fused image without knowing this carrier exists, and the only thing a reader
# gives up is the packed image's exactness.
#
# The alpha is the same dial as encode_dab's, in the same units: the term is
# converted into depth codes here so that one setting looks the same in both
# carriers. A code is DAB_L_PER_CODE = 0.39 L*, so converting at the call site
# would have made the shared default mean two different things.


def encode_fused(
    normals: np.ndarray,
    depth_z: np.ndarray,
    frame: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    z_range: tuple[float, float],
    pack: "NormalDepthPack | None" = None,
    detail_alpha: float = DAB_DETAIL_ALPHA,
    detail_d: int = DAB_DETAIL_D,
    detail_sigma_color: float = DAB_DETAIL_SIGMA_COLOR,
    detail_sigma_space: float = DAB_DETAIL_SIGMA_SPACE,
) -> np.ndarray:
    """Pack normals and depth, with the frame's fine detail over the depth.

    Args:
        normals: (H, W, 3) unit normals, in the model's own frame.
        depth_z: (H, W) the model's depth, in its own units.
        frame: (H, W, 3) uint8 RGB, supplying the luminance the detail is
            taken from. Only its high frequencies are used.
        valid: (H, W) bool, or None to take every representable depth.
        z_range: the ``(z_min, z_max)`` to encode over.
        pack: the codec to build the octahedral pair with, for its pole and
            its conditioning check. Defaults to a plain :class:`NormalDepthPack`.
        detail_alpha: how much of the frame's high-frequency L to lay over the
            depth, in the same units as :func:`encode_dab`'s. Zero gives
            exactly :meth:`NormalDepthPack.encode`'s output.
        detail_d, detail_sigma_color, detail_sigma_space: the bilateral the
            detail is taken against.

    Returns:
        (H, W, 3) uint8 RGB, ``[Oct_U, Oct_V, depth]`` like the packed image.
        Read it with :meth:`NormalDepthPack.decode`; the depth comes back with
        the detail as its error, which is what the detail costs.

    Invalid pixels stay exactly zero: the detail is only added where the
    packed image put a real code, or it would lift the sentinel off zero and
    hand back a perfectly plausible depth for a pixel that has none.
    """
    if pack is None:
        pack = NormalDepthPack()
    plate = pack.encode(normals, depth_z, valid, z_range=z_range)
    if not detail_alpha:
        return plate

    l_star = _rgb_to_lab(
        np.asarray(frame, dtype=np.uint8).astype(np.float32) / 255.0
    )[..., 0]
    delta = detail_alpha * _dab_detail(
        l_star, detail_d, detail_sigma_color, detail_sigma_space
    ) / DAB_L_PER_CODE

    out = plate.copy()
    real = plate[..., 2] > 0
    code = np.where(
        real,
        np.clip(plate[..., 2].astype(np.float32) + delta,
                DEPTH_CODE_MIN, DEPTH_CODE_MAX),
        0.0,
    )
    out[..., 2] = code.astype(np.uint8)
    return out
