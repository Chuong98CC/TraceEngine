"""The MoGe tool's own callables -- the Retinex albedo split behind
``--save-Alb-norm``, and the depth ramp behind ``--save-blend-Dab``.

The split lives in the MoGe tool rather than in ``src/utils``, so these reach
into it; it pulls torch in on import, which is why they are not alongside the
pure-numpy codec tests.

A note on what these can and cannot pin. The failure this split is prone to is
under-smoothing the illumination: too small a ``sigma_r`` leaves the
illumination hugging the input, so the residual is almost pure fine detail, and
the percentile normalization then stretches it -- it spans only ~0.37, so every
step in it is multiplied by ~680x -- until the surface's own noise reads as
speckle. What that looks like is measurable on a real frame but not on a
synthetic one, because the stretch is *relative*: a fixture with only
illumination and no reflectance has nothing to set the scale but the noise, so
it ranks the parameters backwards. The tests below assert the parts that do
survive the trip to synthetic data; the parameter choice itself was settled on
real frames (stretch 683x -> 203x, a flat panel's own noise 3.25x -> 1.68x, and
the illumination dropping from 94% to 39% of the source's high frequencies).
"""

import cv2
import numpy as np

from tools.general_test.module.infer_moge3 import (
    DEFAULT_L_BRIGHT,
    DEFAULT_L_DARK,
    _dab_ramp,
    _scaled_dab_ramp,
    extract_albedo_retinex,
)


def _lit_surface(height: int = 256, width: int = 256, level: float = 110.0,
                 ripple: float = 40.0, weave: float = 12.0,
                 noise: float = 1.0, seed: int = 5):
    """A textured surface under a smooth illumination ripple.

    The weave is what makes this stand in for a photograph: a real frame's high
    frequencies are mostly texture, and a fixture without them measures its own
    noise instead.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    lit = level + ripple * np.sin(np.linspace(0.0, 9.0, width))[None, :]
    texture = weave * np.sin(xx * 1.9) * np.sin(yy * 1.7)
    return (lit.repeat(height, axis=0) + texture
            + rng.normal(0.0, noise, (height, width))).astype(np.float32)


def _high_frequency(x: np.ndarray) -> float:
    """Local variation, in levels: everything a blur of sigma 2 cannot follow."""
    return float((x - cv2.GaussianBlur(x.astype(np.float32), (0, 0), 2.0)).std())


def test_returns_an_albedo_an_illumination_and_the_percentiles():
    """Two uint8 images of the input's shape, and the scaling that was used.

    The percentiles are part of the contract rather than a convenience: the
    albedo they scale between is a Retinex residual sitting near 1, so
    ``p_low < 1 < p_high`` is what says they belong to this image's scaling and
    can be recorded in the sidecar to invert it.
    """
    albedo, illumination, p_low, p_high = extract_albedo_retinex(_lit_surface())

    assert albedo.dtype == np.uint8
    assert albedo.shape == (256, 256)
    assert illumination.dtype == np.uint8
    assert illumination.shape == (256, 256)
    assert p_low < 1.0 < p_high


def test_the_illumination_carries_lighting_rather_than_texture():
    """The illumination estimate must be smoother than the surface it came from.

    This is the defect in numbers: with the original ``sigma_r=0.25`` -- 5% of
    the ~5 log units the image spans -- the bilateral treats nearly every
    texture edge as one worth preserving, and the "illumination" comes back
    carrying 16.7% of the surface's high frequencies instead of 3.9%. On a real
    frame the same mistake reads 94.5% against 39.0%: there it is not a smooth
    lighting map at all, it is a copy of the input.

    The 10% bound is calibrated to this fixture rather than universal -- a real
    frame sits higher, because its high frequencies include mid-scale structure
    an illumination map legitimately tracks. What it guards is the direction:
    an illumination that stops being smooth.
    """
    surface = _lit_surface()

    _, illumination, _, _ = extract_albedo_retinex(surface)

    assert _high_frequency(illumination) <= 0.10 * _high_frequency(surface)


def test_the_pre_filter_can_be_switched_off():
    """``denoise_sigma_s=0`` disables the stage rather than silently no-opping.

    It is the only guard against the pre-filter being dropped or hard-wired,
    which would quietly cost the flat-region noise suppression it exists for.
    """
    surface = _lit_surface()

    filtered, *_ = extract_albedo_retinex(surface)
    plain, *_ = extract_albedo_retinex(surface, denoise_sigma_s=0)

    assert not np.array_equal(filtered, plain)


def test_dab_ramp_resolves_the_gate_and_the_direction():
    """The two ends of the depth ramp, from the gate and the direction flag.

    The gate holds the limits and the flag says which end the *distance* gets,
    so reversing the ramp means flipping one boolean rather than swapping two
    numbers and hoping both call sites noticed. Breaks if the flag stops being
    read, which is the failure that would come back as "the brightness did not
    reverse" with the gate looking perfectly correct.
    """
    l_far, l_near = _dab_ramp(2.0, 95.0, far_is_bright=True)
    assert (l_far, l_near) == (95.0, 2.0)

    l_far, l_near = _dab_ramp(2.0, 95.0, far_is_bright=False)
    assert (l_far, l_near) == (2.0, 95.0)


def test_scaled_dab_ramp_takes_its_ends_from_the_frame():
    """The ramp spans the frame's own luminance, not the fixed gate.

    This is the whole difference from ``--save-blend-Dab``: there the ends are
    the tuning constants whatever the image is, so a low-contrast frame still
    gets a full-range ramp. Breaks if the ends go back to being constants.
    """
    l_star = np.array([[12.0, 40.0], [70.0, 88.0]], dtype=np.float32)

    l_far, l_near = _scaled_dab_ramp(l_star, far_is_bright=True)

    assert (l_far, l_near) == (88.0, 12.0)


def test_scaled_dab_ramp_follows_the_direction_flag():
    """Flipping the flag swaps which end of the frame's range the distance gets."""
    l_star = np.array([[12.0, 88.0]], dtype=np.float32)

    assert _scaled_dab_ramp(l_star, far_is_bright=False) == (12.0, 88.0)


def test_scaled_dab_ramp_keeps_a_valid_pixel_off_the_invalid_sentinel():
    """A near-black frame does not push the ramp's dark end to zero.

    A valid pixel at L* 0 is pure black, which is the sentinel an invalid pixel
    is written as -- the two would be indistinguishable, and the reader would
    report no depth at all for a pixel that has one. The floor sits clear of
    the boundary rather than on it: L* 0.2 already maps to sRGB 1.
    """
    l_star = np.array([[0.0, 60.0]], dtype=np.float32)

    l_far, l_near = _scaled_dab_ramp(l_star, far_is_bright=True)

    assert l_near >= 1.0
    assert l_far == 60.0


def test_scaled_dab_ramp_falls_back_when_the_frame_has_no_range():
    """A frame with no luminance span has nothing to scale to.

    A degenerate ramp would put both ends on the same value, which the encoder
    rejects outright -- so one flat frame would abort the whole run. Falls back
    to the gate instead.
    """
    l_star = np.full((2, 2), 50.0, dtype=np.float32)

    l_far, l_near = _scaled_dab_ramp(l_star, far_is_bright=True)

    # asserted against the constants, not literals: the gate is a tuning knob
    # and this test is about the fallback, not about where the gate sits
    assert (l_far, l_near) == _dab_ramp(DEFAULT_L_DARK, DEFAULT_L_BRIGHT, True)
