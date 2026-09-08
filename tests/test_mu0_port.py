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


def test_config_encode_decode_roundtrip():
    cfg_mod = importlib.import_module("mu0.policies.configuration_smolvla")
    SmolVLAConfig = cfg_mod.SmolVLAConfig
    cfg = SmolVLAConfig(trace_mode=True, num_vlm_layers=20, depth_clone_stem=True)
    import draccus

    raw = draccus.encode(cfg)
    dec = draccus.decode(SmolVLAConfig, raw)  # draccus 0.8: decode(target_type, raw)
    assert dec.trace_mode is True
    assert dec.num_vlm_layers == 20
    assert dec.depth_clone_stem is True


def test_from_ckpt_config_release():
    cfg_mod = importlib.import_module("mu0.policies.configuration_smolvla")
    ckpt = MU0_RELEASE / "final_ckpt"
    if not (ckpt / "config.json").exists():
        pytest.skip("release checkpoint not present")
    cfg = cfg_mod.from_ckpt_config(ckpt)
    assert cfg.trace_mode is True
    assert cfg.vlm_model_name.endswith("SmolVLM2-2.2B-Instruct")
    assert cfg.num_vlm_layers == 20
    assert cfg.num_expert_layers == 20
    assert cfg.expert_width_multiplier == 0.5
    assert cfg.depth_clone_stem is True
    assert cfg.depth_lora_rank == 8
    assert cfg.use_dino is True
    assert cfg.trace_bspline_n_ctrl == 10
    assert cfg.history_len == 8 and cfg.future_len == 32
    # Field is tuple[int, int]; draccus decodes the JSON list into a tuple.
    assert cfg.resize_imgs_with_padding == (512, 512)
