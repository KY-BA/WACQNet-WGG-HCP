"""Background-tail-invariant Group-DRO score normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class GDROFit:
    method: str
    feature_names: tuple[str, ...]
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    correction_center: float
    objective: float
    data_component: float
    tail_variance: float
    tail_cvar: float
    l2_component: float
    optimizer_success: bool
    optimizer_message: str

    def correction(self, frame: pd.DataFrame, clip: float) -> np.ndarray:
        if not self.feature_names:
            return np.zeros(len(frame), dtype=float)
        x = frame[list(self.feature_names)].to_numpy(float)
        z = (x - np.asarray(self.feature_center)) / np.asarray(self.feature_scale)
        raw = z @ np.asarray(self.coefficients) - float(self.correction_center)
        return np.clip(raw, -float(clip), float(clip))

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "feature_names": list(self.feature_names),
            "feature_center": list(self.feature_center),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "coefficient_map": dict(zip(self.feature_names, self.coefficients)),
            "correction_center": self.correction_center,
            "objective": self.objective,
            "data_component": self.data_component,
            "tail_variance": self.tail_variance,
            "tail_cvar": self.tail_cvar,
            "l2_component": self.l2_component,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
        }


def background_equal_weights(groups: pd.Series) -> np.ndarray:
    counts = groups.map(groups.value_counts()).to_numpy(float)
    weights = 1.0 / counts
    return weights / weights.sum()


def prepare_gdro_features(frame: pd.DataFrame, mean_power_b: float, eps: float = 1e-8) -> pd.DataFrame:
    required = {"background_id", "mu_fixed", "absolute_error", "background_sigma", "valid_fraction", "retained_valid_fraction", "input_wind_speed"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing BTI-GDRO fields: {sorted(missing)}")
    result = frame.copy()
    mu = np.maximum(result.mu_fixed.to_numpy(float), 0.0)
    valid = np.clip(result.valid_fraction.to_numpy(float), eps, 1.0 - eps)
    retained = np.clip(result.retained_valid_fraction.to_numpy(float), eps, 1.0)
    wind = np.maximum(result.input_wind_speed.to_numpy(float), eps)
    result["log_mu"] = np.log(mu + 1.0)
    result["log_background_sigma"] = np.log(np.maximum(result.background_sigma.to_numpy(float), eps))
    result["logit_valid_fraction"] = np.log(valid / (1.0 - valid))
    result["negative_log_retained_fraction"] = -np.log(retained)
    result["negative_log_input_wind_speed"] = -np.log(wind)
    result["base_scale"] = (mu + 1.0) ** float(mean_power_b)
    result["log_base_score"] = np.log(result.absolute_error.to_numpy(float) + 1e-3) - np.log(result.base_scale.to_numpy(float))
    return result


def _components(beta: np.ndarray, z: np.ndarray, y: np.ndarray, groups: np.ndarray, weights: np.ndarray, *, data_loss: str, tail_quantile: float, variance_weight: float, cvar_weight: float, cvar_fraction: float, l2_penalty: float) -> tuple[float, float, float, float, float, float]:
    raw = z @ beta
    correction_center = float(np.sum(weights * raw))
    residual = y - (raw - correction_center)
    if data_loss == "huber":
        absolute = np.abs(residual)
        per_row = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
    elif data_loss == "pinball90":
        tau = float(tail_quantile)
        per_row = np.maximum(tau * residual, (tau - 1.0) * residual)
    else:
        per_row = np.zeros_like(residual)
    data_component = float(np.sum(weights * per_row))
    tails = np.asarray([np.quantile(residual[groups == group], tail_quantile) for group in sorted(pd.unique(groups))], float)
    centered = tails - float(np.median(tails))
    tail_variance = float(np.var(tails, ddof=0))
    n_cvar = max(1, int(np.ceil(float(cvar_fraction) * len(tails))))
    tail_cvar = float(np.mean(np.sort(np.abs(centered))[-n_cvar:]))
    l2 = float(np.sum(beta**2))
    total = data_component + float(variance_weight) * tail_variance + float(cvar_weight) * tail_cvar + float(l2_penalty) * l2
    return total, data_component, tail_variance, tail_cvar, l2, correction_center


def fit_gdro(frame: pd.DataFrame, *, method: str, feature_names: Sequence[str], data_loss: str, tail_quantile: float, variance_weight: float, cvar_weight: float, cvar_fraction: float, l2_penalty: float, coefficient_bounds: tuple[float, float], maxiter: int) -> GDROFit:
    names = tuple(feature_names)
    x = frame[list(names)].to_numpy(float)
    groups = frame.background_id.astype(str).to_numpy()
    weights = background_equal_weights(frame.background_id)
    center = np.sum(weights[:, None] * x, axis=0)
    variance = np.sum(weights[:, None] * (x - center) ** 2, axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    z = (x - center) / scale
    y = frame.log_base_score.to_numpy(float)
    lower, upper = map(float, coefficient_bounds)

    def objective(beta: np.ndarray) -> float:
        return _components(beta, z, y, groups, weights, data_loss=data_loss, tail_quantile=tail_quantile, variance_weight=variance_weight, cvar_weight=cvar_weight, cvar_fraction=cvar_fraction, l2_penalty=l2_penalty)[0]

    result = minimize(objective, np.zeros(len(names), dtype=float), method="Powell", bounds=[(lower, upper)] * len(names), options={"maxiter": int(maxiter), "xtol": 1e-4, "ftol": 1e-6})
    total, data_component, tail_variance, tail_cvar, l2, correction_center = _components(result.x, z, y, groups, weights, data_loss=data_loss, tail_quantile=tail_quantile, variance_weight=variance_weight, cvar_weight=cvar_weight, cvar_fraction=cvar_fraction, l2_penalty=l2_penalty)
    return GDROFit(method, names, tuple(float(v) for v in center), tuple(float(v) for v in scale), tuple(float(v) for v in result.x), correction_center, total, data_component, tail_variance, tail_cvar, l2, bool(result.success), str(result.message))


def deployable_scale(frame: pd.DataFrame, fit: GDROFit, correction_clip: float) -> np.ndarray:
    scale = frame.base_scale.to_numpy(float) * np.exp(fit.correction(frame, correction_clip))
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError("Invalid BTI-GDRO deployable scale")
    return scale
