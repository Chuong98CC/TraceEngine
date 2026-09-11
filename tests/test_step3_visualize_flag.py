"""CPU-only unit tests for the merged Step-3 visualization flag.

Step 3 used to switch visualization per artefact: Step 3b had ``--no-viz``
(skip viz.png) and ``--viz-motion-union`` (add union_mask.png), Step 3a'
had its own ``--visualize`` for a debug flow.png, and the driver exposed
the pair as ``--no-viz`` / ``--visualize-motion``. It is now one
``--visualize`` per entry point:

- Step 3b renders viz.png **and** — for a manipulator whose mask the
  flow rescue widened — union_mask.png;
- Step 3a' has no visualization left: flow.png is gone, the motion mask
  surfacing only where it matters, in 3b's union;
- the drivers (run_step3_init_points.py, run_e2e_init_points.py)
  forward ``--visualize`` to Step 3b alone.

Off everywhere by default: nothing is written unless ``--visualize``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.astribot import run_step3_init_points as step3_driver
from tools.general_test.pipeline.run_object_init_points import (
    parse_args as init_points_parse_args,
)

REPO_ROOT = Path("/repo")


def _driver_args(*extra: str):
    """parse_args without a program name (argparse's argv excludes it)."""
    return step3_driver.parse_args(["--repo-id", "r", "--data-root", "/tmp/x",
                                    *extra])


# --- Step-3 driver ------------------------------------------------------------


def test_driver_visualize_is_off_by_default():
    assert _driver_args().visualize is False


def test_driver_forwards_visualize_to_3b_only():
    args = _driver_args("--visualize")
    assert args.visualize is True
    cmd = step3_driver._build_3b_cmd(args, repo_root=REPO_ROOT)
    assert cmd.count("--visualize") == 1
    assert "--no-viz" not in cmd and "--viz-motion-union" not in cmd
    # Step 3a' writes its motion_rle.json and nothing else
    motion = step3_driver._build_motion_masks_cmd(args, repo_root=REPO_ROOT)
    assert "--visualize" not in motion


def test_driver_sends_no_viz_flag_without_visualize():
    cmd = step3_driver._build_3b_cmd(_driver_args(), repo_root=REPO_ROOT)
    assert "--visualize" not in cmd
    assert "--no-viz" not in cmd and "--viz-motion-union" not in cmd


def test_driver_drops_the_merged_flags():
    for flag in ("--no-viz", "--visualize-motion"):
        with pytest.raises(SystemExit):
            _driver_args(flag)


# --- Step 3b ------------------------------------------------------------------


def test_3b_visualize_is_off_by_default():
    assert init_points_parse_args([]).visualize is False


def test_3b_visualize_turns_the_rendering_on():
    assert init_points_parse_args(["--visualize"]).visualize is True


def test_3b_drops_the_merged_flags():
    for flag in ("--no-viz", "--viz-motion-union"):
        with pytest.raises(SystemExit):
            init_points_parse_args([flag])
