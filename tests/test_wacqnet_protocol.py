from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import yaml

from src.wacqnet import build_spatial_features, build_wacqnet, sample_loss


CFG_PATH = Path("D:/CO2/_third/configs/wacqnet_development_protocol_v1.yaml")


def config():
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


def test_protocol_is_frozen_before_training():
    cfg = config()
    audit = json.loads((Path(cfg["outputs"]["protocol_dir"]) / "protocol_freeze_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["protocol_version"] == "1.0.2"
    assert audit["backgrounds"] == 36


def test_current_calibration_and_test_forbidden():
    cfg = config()
    assert cfg["data"]["current_v3_calibration_allowed"] is False
    assert cfg["data"]["current_v3_confirmatory_test_allowed"] is False
    assert cfg["data"]["all_historical_confirmatory_tests_allowed"] is False


def test_primary_candidate_fixed_and_ablations_cannot_replace_it():
    cfg = config()
    assert cfg["ablations"]["primary_candidate"] == "WACQ_FULL"
    assert cfg["ablations"]["ablations_can_replace_primary"] is False
    assert {x["id"] for x in cfg["ablations"]["trained"]} == {
        "WACQ_NO_WIND", "WACQ_NO_CONTRAST", "WACQ_NO_GDRO", "WACQ_FULL"
    }


def test_lobo_is_background_level_and_heldout_is_never_training_data():
    val = config()["validation"]
    assert val["folds"] == 36
    assert val["training_groups_per_fold"] == 35
    assert val["heldout_groups_per_fold"] == 1
    assert val["heldout_used_for_training"] is False
    assert val["heldout_used_for_early_stopping"] is False
    assert val["heldout_used_for_hyperparameter_selection"] is False


def test_levelb_seed_and_training_seed_are_separate():
    cfg = config()
    assert cfg["level_b"]["generation_seed"] == 42
    assert cfg["training"]["training_seed"] == 20260927


def test_model_fit_and_lobo_qhat_rows_are_disjoint():
    train = set(config()["training"]["training_repetitions_per_condition"])
    calibrate = set(config()["training"]["lobo_calibration_repetitions_per_condition"])
    assert train == {0, 25}
    assert calibrate == {1, 13, 26, 38}
    assert train.isdisjoint(calibrate)


def test_spatial_features_are_deployable_and_finite():
    rng = np.random.default_rng(2)
    image = np.zeros((32, 32, 3), np.float32)
    image[..., 0] = 410 + rng.normal(size=(32, 32))
    image[..., 1] = 3.0
    image[..., 2] = 4.0
    valid = rng.random((32, 32)) > 0.1
    image[~valid, 0] = np.nan
    template = np.exp(-((np.arange(32)[None, :] - 16) ** 2) / 20) * np.ones((32, 1))
    spatial, observed = build_spatial_features(image, valid, template, (16.0, 16.0))
    assert spatial.shape == (16, 16, 6)
    assert observed.shape == (2,)
    assert np.isfinite(spatial).all()
    assert np.isfinite(observed).all()


def test_model_outputs_nonnegative_ordered_scales_and_anchor_identity_at_initialization():
    cfg = config()
    full = next(x for x in cfg["ablations"]["trained"] if x["id"] == "WACQ_FULL")
    model = build_wacqnet(cfg, full)
    spatial = np.zeros((3, 16, 16, 6), np.float32)
    scalar = np.asarray([[np.log1p(1.0), 1.0, 0.8], [np.log1p(10.0), 1.0, 0.8], [np.log1p(40.0), 1.0, 0.8]], np.float32)
    out = model([spatial, scalar], training=False)
    assert np.allclose(out["mu"].numpy().reshape(-1), [1.0, 10.0, 40.0], atol=1e-5)
    assert (out["s_minus"].numpy() > 0).all()
    assert (out["s_plus"].numpy() > 0).all()


def test_loss_is_per_sample_and_finite():
    outputs = {
        "mu": tf.constant([[4.0], [9.0]]),
        "s_minus": tf.constant([[2.0], [3.0]]),
        "s_plus": tf.constant([[3.0], [4.0]]),
        "delta": tf.constant([[0.1], [-0.2]]),
    }
    value = sample_loss(tf.constant([5.0, 10.0]), outputs)
    assert value.shape == (2,)
    assert np.isfinite(value.numpy()).all()
