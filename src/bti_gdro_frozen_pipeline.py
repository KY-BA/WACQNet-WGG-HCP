"""Frozen deployable A4 BTI-GDRO-HCP interval pipeline helpers."""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from src.bti_gdro import GDROFit
from src.bti_hcp import hierarchical_conformal_quantile


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_from_artifact(artifact: dict[str, Any]) -> GDROFit:
    payload = artifact["fit"]
    names = {field.name for field in fields(GDROFit)}
    values = {key: payload[key] for key in names}
    for key in ("feature_names", "feature_center", "feature_scale", "coefficients"):
        values[key] = tuple(values[key])
    return GDROFit(**values)


def input_wind_speed(frame: pd.DataFrame) -> np.ndarray:
    """Reconstruct the wind speed supplied to frozen HPR from condition metadata."""
    result = frame.true_wind_speed.to_numpy(float).copy()
    speed_error = frame.experiment.astype(str).eq("wind_speed_input_error").to_numpy()
    parameter = frame.condition_id.astype(str).str.extract(
        r"\|parameter=([-+0-9.eE]+)(?:\||$)"
    )[0]
    factor = pd.to_numeric(parameter, errors="coerce").to_numpy(float)
    if np.isnan(factor[speed_error]).any():
        raise RuntimeError("Cannot reconstruct input wind speed")
    result[speed_error] *= factor[speed_error]
    if not np.isfinite(result).all() or (result <= 0).any():
        raise RuntimeError("Invalid reconstructed input wind speed")
    return result


def deployable_features(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    mean_power_b: float,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Build A4 features without accessing true emission or residuals."""
    required = {
        "background_id", "mu_fixed", "valid_fraction", "retained_valid_fraction",
        "true_wind_speed", "experiment", "condition_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing deployable A4 fields: {sorted(missing)}")
    metadata = manifest.set_index("background_id")
    result = frame.copy()
    result["background_sigma"] = result.background_id.map(metadata.background_sigma)
    if result.background_sigma.isna().any():
        raise RuntimeError("Missing background_sigma for A4 deployment")
    result["input_wind_speed"] = input_wind_speed(result)
    mu = np.maximum(result.mu_fixed.to_numpy(float), 0.0)
    valid = np.clip(result.valid_fraction.to_numpy(float), epsilon, 1.0 - epsilon)
    retained = np.clip(
        result.retained_valid_fraction.to_numpy(float), epsilon, 1.0
    )
    wind = np.maximum(result.input_wind_speed.to_numpy(float), epsilon)
    result["log_mu"] = np.log(mu + 1.0)
    result["log_background_sigma"] = np.log(
        np.maximum(result.background_sigma.to_numpy(float), epsilon)
    )
    result["logit_valid_fraction"] = np.log(valid / (1.0 - valid))
    result["negative_log_retained_fraction"] = -np.log(retained)
    result["negative_log_input_wind_speed"] = -np.log(wind)
    result["base_scale"] = (mu + 1.0) ** float(mean_power_b)
    return result


def apply_frozen_scale(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    mean_power_b: float,
    correction_clip: float,
) -> pd.DataFrame:
    result = deployable_features(
        frame, manifest, mean_power_b=mean_power_b
    )
    fit = fit_from_artifact(artifact)
    x = result[list(fit.feature_names)].to_numpy(float)
    z = (x - np.asarray(fit.feature_center)) / np.asarray(fit.feature_scale)
    # Algebraically identical to ``z @ beta``.  The explicit column sum avoids
    # a Windows BLAS delay-load failure observed only after TensorFlow inference.
    raw = np.zeros(len(result), dtype=float)
    for column, coefficient in enumerate(fit.coefficients):
        raw += z[:, column] * float(coefficient)
    correction = np.clip(
        raw - float(fit.correction_center), -float(correction_clip), float(correction_clip)
    )
    result["gdro_correction"] = correction
    result["normalizer_scale"] = result.base_scale.to_numpy(float) * np.exp(correction)
    if (
        not np.isfinite(result.normalizer_scale.to_numpy(float)).all()
        or (result.normalizer_scale.to_numpy(float) <= 0).any()
    ):
        raise RuntimeError("Invalid frozen A4 deployable scale")
    return result


def add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add truth-dependent calibration scores after deployable scale is fixed."""
    result = frame.copy()
    result["absolute_error"] = np.abs(
        result.true_emission.to_numpy(float) - result.mu_fixed.to_numpy(float)
    )
    result["ape"] = result.absolute_error / result.true_emission
    result["score"] = result.absolute_error / result.normalizer_scale
    if not np.isfinite(result.score).all():
        raise RuntimeError("Non-finite A4 nonconformity score")
    return result


def hcp_qhat(frame: pd.DataFrame, coverage: float) -> float:
    groups = [
        part.score.to_numpy(float)
        for _, part in frame.groupby("background_group_id", sort=True)
    ]
    return hierarchical_conformal_quantile(groups, coverage)


def add_intervals(frame: pd.DataFrame, qhat: float) -> pd.DataFrame:
    result = frame.copy()
    half = float(qhat) * result.normalizer_scale.to_numpy(float)
    result["lower90"] = np.maximum(0.0, result.mu_fixed.to_numpy(float) - half)
    result["upper90"] = result.mu_fixed.to_numpy(float) + half
    result["width90"] = result.upper90 - result.lower90
    result["covered90"] = (
        (result.true_emission >= result.lower90)
        & (result.true_emission <= result.upper90)
    )
    return result


def summarize(part: pd.DataFrame, nominal: float = 0.90) -> dict[str, float | int]:
    covered = part.covered90.to_numpy(bool)
    width = part.width90.to_numpy(float)
    picp = float(covered.mean())
    normalized = width / (np.maximum(part.mu_fixed.to_numpy(float), 0.0) + 1.0)
    return {
        "n": int(len(part)),
        "picp90": picp,
        "coverage_error": abs(picp - float(nominal)),
        "mpiw": float(width.mean()),
        "median_width": float(np.median(width)),
        "p90_width": float(np.quantile(width, 0.90)),
        "p95_width": float(np.quantile(width, 0.95)),
        "p99_width": float(np.quantile(width, 0.99)),
        "max_width": float(np.max(width)),
        "fraction_width_gt_50": float(np.mean(width > 50.0)),
        "fraction_width_gt_100": float(np.mean(width > 100.0)),
        "lower_zero_fraction": float(np.mean(part.lower90.to_numpy(float) == 0.0)),
        "median_normalized_width": float(np.median(normalized)),
        "median_ape": float(np.median(part.ape.to_numpy(float))),
        "p_reliable30": float(np.mean(part.ape.to_numpy(float) < 0.30)),
    }
