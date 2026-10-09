"""Asymmetric background-tail-invariant Group-DRO scale learning.

The module learns two deployable positive scales around a frozen point
prediction: one for over-prediction and one for under-prediction.  Truth is
used only as a Development target; deployment scales depend exclusively on
observable features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class AsymmetricFit:
    method: str
    feature_names: tuple[str, ...]
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients_minus: tuple[float, ...]
    coefficients_plus: tuple[float, ...]
    target_center_minus: float
    target_center_plus: float
    objective: float
    erm_component: float
    gdro_component: float
    tail_variance: float
    tail_cvar: float
    width_component: float
    l2_component: float
    optimizer_success: bool
    optimizer_message: str

    def to_dict(self) -> dict[str, object]:
        scale = np.asarray(self.feature_scale, float)
        minus = np.asarray(self.coefficients_minus, float)
        plus = np.asarray(self.coefficients_plus, float)
        return {
            "method": self.method,
            "feature_names": list(self.feature_names),
            "feature_center": list(self.feature_center),
            "feature_scale": list(self.feature_scale),
            "coefficients_minus": list(self.coefficients_minus),
            "coefficients_plus": list(self.coefficients_plus),
            "coefficient_map_minus_raw": dict(zip(self.feature_names, minus / scale)),
            "coefficient_map_plus_raw": dict(zip(self.feature_names, plus / scale)),
            "target_center_minus": self.target_center_minus,
            "target_center_plus": self.target_center_plus,
            "objective": self.objective,
            "erm_component": self.erm_component,
            "gdro_component": self.gdro_component,
            "tail_variance": self.tail_variance,
            "tail_cvar": self.tail_cvar,
            "width_component": self.width_component,
            "l2_component": self.l2_component,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
        }


def background_equal_weights(groups: pd.Series) -> np.ndarray:
    counts = groups.astype(str).map(groups.astype(str).value_counts()).to_numpy(float)
    weights = 1.0 / counts
    return weights / weights.sum()


def prepare_asymmetric_features(frame: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    required = {
        "background_id", "true_emission", "mu_fixed", "background_sigma",
        "valid_fraction", "retained_valid_fraction", "input_wind_speed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing asymmetric BTI fields: {sorted(missing)}")
    result = frame.copy()
    truth = result.true_emission.to_numpy(float)
    mu = result.mu_fixed.to_numpy(float)
    valid = np.clip(result.valid_fraction.to_numpy(float), eps, 1.0 - eps)
    retained = np.clip(result.retained_valid_fraction.to_numpy(float), eps, 1.0)
    wind = np.maximum(result.input_wind_speed.to_numpy(float), eps)
    result["absolute_error"] = np.abs(truth - mu)
    result["over_error"] = np.maximum(mu - truth, 0.0)
    result["under_error"] = np.maximum(truth - mu, 0.0)
    result["log_mu"] = np.log(np.maximum(mu, 0.0) + 1.0)
    result["log_background_sigma"] = np.log(
        np.maximum(result.background_sigma.to_numpy(float), eps)
    )
    result["logit_valid_fraction"] = np.log(valid / (1.0 - valid))
    result["negative_log_retained_fraction"] = -np.log(retained)
    result["negative_log_input_wind_speed"] = -np.log(wind)
    return result


def _pinball(residual: np.ndarray, tau: float) -> np.ndarray:
    return np.maximum(float(tau) * residual, (float(tau) - 1.0) * residual)


def _logmeanexp(values: np.ndarray, temperature: float) -> float:
    scaled = np.asarray(values, float) / float(temperature)
    maximum = float(np.max(scaled))
    return float(temperature) * (maximum + float(np.log(np.mean(np.exp(scaled - maximum)))))


def _weighted_target_center(values: np.ndarray, groups: np.ndarray) -> float:
    series = pd.Series(groups)
    weights = background_equal_weights(series)
    return float(np.sum(weights * values))


def _components(
    parameters: np.ndarray,
    z: np.ndarray,
    frame: pd.DataFrame,
    *,
    method: str,
    tau: float,
    dro_temperature: float,
    tail_quantile: float,
    tail_variance_weight: float,
    tail_cvar_weight: float,
    tail_cvar_fraction: float,
    width_weight: float,
    l2_penalty: float,
    correction_clip: float,
    residual_epsilon: float,
) -> tuple[float, dict[str, float]]:
    n_features = z.shape[1]
    beta_minus = parameters[:n_features]
    beta_plus = parameters[n_features:]
    log_minus = np.clip(z @ beta_minus, -correction_clip, correction_clip)
    log_plus = np.clip(z @ beta_plus, -correction_clip, correction_clip)
    groups = frame.background_id.astype(str).to_numpy()
    over = frame.over_error.to_numpy(float)
    under = frame.under_error.to_numpy(float)
    minus_mask, plus_mask = over > 0.0, under > 0.0
    if not minus_mask.any() or not plus_mask.any():
        raise RuntimeError("Both residual directions are required for asymmetric fitting")

    y_minus = np.log(over[minus_mask] + residual_epsilon)
    y_plus = np.log(under[plus_mask] + residual_epsilon)
    center_minus = _weighted_target_center(y_minus, groups[minus_mask])
    center_plus = _weighted_target_center(y_plus, groups[plus_mask])
    loss_minus = _pinball((y_minus - center_minus) - log_minus[minus_mask], tau)
    loss_plus = _pinball((y_plus - center_plus) - log_plus[plus_mask], tau)

    group_losses: list[float] = []
    for group in sorted(pd.unique(groups)):
        gm = groups[minus_mask] == group
        gp = groups[plus_mask] == group
        pieces = []
        if gm.any():
            pieces.append(float(np.mean(loss_minus[gm])))
        if gp.any():
            pieces.append(float(np.mean(loss_plus[gp])))
        group_losses.append(float(np.mean(pieces)))
    group_losses_array = np.asarray(group_losses, float)
    erm = float(np.mean(group_losses_array))
    gdro = _logmeanexp(group_losses_array, dro_temperature)

    scale_minus = np.exp(log_minus)
    scale_plus = np.exp(log_plus)
    score = np.maximum(over / scale_minus, under / scale_plus)
    log_score = np.log(score + residual_epsilon)
    tails = np.asarray([
        np.quantile(log_score[groups == group], tail_quantile)
        for group in sorted(pd.unique(groups))
    ])
    tail_variance = float(np.var(tails, ddof=0))
    centered_tail = np.abs(tails - float(np.median(tails)))
    n_cvar = max(1, int(np.ceil(tail_cvar_fraction * len(tails))))
    tail_cvar = float(np.mean(np.sort(centered_tail)[-n_cvar:]))
    row_weights = background_equal_weights(frame.background_id)
    width_component = float(np.sum(row_weights * (scale_minus + scale_plus)))
    l2 = float(np.sum(parameters**2))

    if method == "A1_ASYMMETRIC_MEAN_POWER":
        data_component = erm
    else:
        data_component = gdro
    total = (
        data_component
        + tail_variance_weight * tail_variance
        + tail_cvar_weight * tail_cvar
        + width_weight * width_component
        + l2_penalty * l2
    )
    values = {
        "erm": erm,
        "gdro": gdro,
        "tail_variance": tail_variance,
        "tail_cvar": tail_cvar,
        "width": width_component,
        "l2": l2,
        "target_center_minus": center_minus,
        "target_center_plus": center_plus,
    }
    return float(total), values


def fit_asymmetric_gdro(
    frame: pd.DataFrame,
    *,
    method: str,
    feature_names: Sequence[str],
    tau: float,
    dro_temperature: float,
    tail_quantile: float,
    tail_variance_weight: float,
    tail_cvar_weight: float,
    tail_cvar_fraction: float,
    width_weight: float,
    l2_penalty: float,
    log_mu_bounds: tuple[float, float],
    other_bounds: tuple[float, float],
    correction_clip: float,
    residual_epsilon: float,
    maxiter: int,
) -> AsymmetricFit:
    names = tuple(feature_names)
    x = frame[list(names)].to_numpy(float)
    weights = background_equal_weights(frame.background_id)
    center = np.sum(weights[:, None] * x, axis=0)
    variance = np.sum(weights[:, None] * (x - center) ** 2, axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    z = (x - center) / scale

    standardized_bounds: list[tuple[float, float]] = []
    for name, feature_scale in zip(names, scale):
        raw_bounds = log_mu_bounds if name == "log_mu" else other_bounds
        standardized_bounds.append(tuple(float(bound) * float(feature_scale) for bound in raw_bounds))
    bounds = standardized_bounds + standardized_bounds

    def objective(parameters: np.ndarray) -> float:
        return _components(
            parameters, z, frame, method=method, tau=tau,
            dro_temperature=dro_temperature, tail_quantile=tail_quantile,
            tail_variance_weight=tail_variance_weight,
            tail_cvar_weight=tail_cvar_weight,
            tail_cvar_fraction=tail_cvar_fraction, width_weight=width_weight,
            l2_penalty=l2_penalty, correction_clip=correction_clip,
            residual_epsilon=residual_epsilon,
        )[0]

    result = minimize(
        objective,
        np.zeros(2 * len(names), dtype=float),
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(maxiter), "xtol": 1e-4, "ftol": 1e-6},
    )
    total, parts = _components(
        result.x, z, frame, method=method, tau=tau,
        dro_temperature=dro_temperature, tail_quantile=tail_quantile,
        tail_variance_weight=tail_variance_weight,
        tail_cvar_weight=tail_cvar_weight,
        tail_cvar_fraction=tail_cvar_fraction, width_weight=width_weight,
        l2_penalty=l2_penalty, correction_clip=correction_clip,
        residual_epsilon=residual_epsilon,
    )
    n_features = len(names)
    return AsymmetricFit(
        method=method,
        feature_names=names,
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        coefficients_minus=tuple(float(value) for value in result.x[:n_features]),
        coefficients_plus=tuple(float(value) for value in result.x[n_features:]),
        target_center_minus=parts["target_center_minus"],
        target_center_plus=parts["target_center_plus"],
        objective=total,
        erm_component=parts["erm"],
        gdro_component=parts["gdro"],
        tail_variance=parts["tail_variance"],
        tail_cvar=parts["tail_cvar"],
        width_component=parts["width"],
        l2_component=parts["l2"],
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


def asymmetric_scales(
    frame: pd.DataFrame,
    fit: AsymmetricFit,
    correction_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = frame[list(fit.feature_names)].to_numpy(float)
    z = (x - np.asarray(fit.feature_center)) / np.asarray(fit.feature_scale)
    log_minus = np.clip(
        z @ np.asarray(fit.coefficients_minus), -correction_clip, correction_clip
    )
    log_plus = np.clip(
        z @ np.asarray(fit.coefficients_plus), -correction_clip, correction_clip
    )
    minus, plus = np.exp(log_minus), np.exp(log_plus)
    if not np.isfinite(minus).all() or not np.isfinite(plus).all():
        raise RuntimeError("Non-finite asymmetric scale")
    if (minus <= 0).any() or (plus <= 0).any():
        raise RuntimeError("Non-positive asymmetric scale")
    return minus, plus


def asymmetric_scores(
    frame: pd.DataFrame,
    scale_minus: np.ndarray,
    scale_plus: np.ndarray,
) -> np.ndarray:
    truth = frame.true_emission.to_numpy(float)
    mu = frame.mu_fixed.to_numpy(float)
    return np.maximum(
        np.maximum(mu - truth, 0.0) / np.asarray(scale_minus, float),
        np.maximum(truth - mu, 0.0) / np.asarray(scale_plus, float),
    )
