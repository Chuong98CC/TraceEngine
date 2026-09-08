"""CPU-only checks for the src/mu0 port (see docs/superpowers/plans/2026-09-07-mu0-port.md)."""
import json
import importlib
from pathlib import Path

import pytest

MU0_RELEASE = Path("/home/chuong/workspace/point_models/mu0")

ALL_PORTED_MODULES = [
    "mu0.datasets.bspline_basis",
    "mu0.datasets.trace_depth",
    "mu0.datasets.trace_delta_stats",
    "mu0.datasets.trace_dataset",
    "mu0.policies.configuration_smolvla",
    "mu0.policies.smolvlm_with_expert",
    "mu0.policies.modeling_smolvla",
    "mu0.policies.processor_smolvla",
    "mu0.policies.visualize_trace",
    "mu0.scripts.lerobot_train_trace_mu0",
    "mu0.scripts.lerobot_predict_trace_mu0_image_only",
]


def test_all_port_modules_import():
    for name in ALL_PORTED_MODULES:
        importlib.import_module(name)


def test_stats_load_release():
    load_trace_stats = importlib.import_module(
        "mu0.datasets.trace_delta_stats"
    ).load_trace_stats
    p = MU0_RELEASE / "normalizer_stats.json"
    if not p.exists():
        pytest.skip("release normalizer_stats.json not present")
    stats = load_trace_stats(str(p))
    assert stats.target_kind == "anchor"


def test_stats_module_runnable():
    # `python -m mu0.datasets.trace_delta_stats --help` must exit 0 (module has a CLI).
    import subprocess, sys

    subprocess.run(
        [sys.executable, "-m", "mu0.datasets.trace_delta_stats", "--help"],
        check=True,
        capture_output=True,
    )
