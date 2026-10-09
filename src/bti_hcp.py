"""Background-tail-invariant scale learning for grouped conformal prediction.

This module contains no HPR or V2 inference.  It operates only on frozen
development predictions and deployable observation-side features.  The score
normalizer is learned with background-equal weighting and a penalty on the
between-background dispersion of upper-tail normalized residuals.

The hierarchical *coverage* theorem is deliberately not claimed here.  This
module implements the development-side scale learner and background-crossfit
evaluation needed before mapping the final calibration rule to a published
hierarchical conformal procedure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


FEATURE_COLUMNS = (
    "log_mu",
    "negative_log_template_energy_retained",
    "log_background_sigma",
    "negative_log_template_mass_retained",
    "negative_log_input_wind_speed",
)


def stable_seed(seed: int, *parts: object) -> int:
    token = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def background_equal_weights(background_ids: Iterable[object]) -> np.ndarray:
    values = pd.Series(list(background_ids), dtype="object")
    counts = values.map(values.value_counts()).to_numpy(float)
    weights = 1.0 / counts
    return weights / weights.mean()


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> tuple[float, int]:
    values = np.asarray(scores, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite conformal scores")
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    rank = min(int(math.ceil((len(values) + 1) * float(coverage))), len(values))
    return float(np.partition(values, rank - 1)[rank - 1]), rank


def hierarchical_conformal_quantile(group_scores: Sequence[np.ndarray], coverage: float) -> float:
    """HCP threshold from Lee, Barber & Willett (2025), Equation (6).

    Every calibration group receives total mass 1/(K+1), divided equally
    across observations in that group, and +infinity receives mass 1/(K+1).
    A finite 90% threshold therefore requires at least nine calibration groups.
    """
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    groups = [np.asarray(scores, float) for scores in group_scores]
    if not groups or any(group.size == 0 for group in groups):
        raise ValueError("Each HCP calibration group must contain finite scores")
    groups = [group[np.isfinite(group)] for group in groups]
    if any(group.size == 0 for group in groups):
        raise ValueError("Each HCP calibration group must contain finite scores")
    k_groups = len(groups)
    finite_mass = k_groups / (k_groups + 1.0)
    if float(coverage) > finite_mass + 1e-15:
        return float("inf")
    values = np.concatenate(groups)
    weights = np.concatenate([
        np.full(group.size, 1.0 / ((k_groups + 1.0) * group.size), dtype=float)
        for group in groups
    ])
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, float(coverage), side="left"))
    return float(values[order[min(position, len(order) - 1)]])


def minimum_hcp_groups_for_finite_threshold(coverage: float) -> int:
    """Smallest K for which K/(K+1) is at least the target coverage."""
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    return int(math.ceil(float(coverage) / (1.0 - float(coverage)) - 1e-12))


def balanced_scores(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, bool, int]:
    """Return an equal-per-background deterministic score sample."""
    counts = frame.groupby("background_id", sort=True).size()
    if counts.empty:
        raise ValueError("No background scores")
    n_equal = int(counts.min())
    if counts.nunique() == 1:
        return frame.score.to_numpy(float), False, n_equal
    parts: list[np.ndarray] = []
    for background, part in frame.groupby("background_id", sort=True):
        rng = np.random.default_rng(stable_seed(seed, background))
        positions = rng.choice(len(part), size=n_equal, replace=False)
        parts.append(part.score.to_numpy(float)[positions])
    return np.concatenate(parts), True, n_equal


def equal_background_subsample(frame: pd.DataFrame, rows_per_background: int, seed: int) -> pd.DataFrame:
    """Deterministically cap fit rows while preserving equal background weight."""
    parts: list[pd.DataFrame] = []
    for background, part in frame.groupby("background_id", sort=True):
        n = min(int(rows_per_background), len(part))
        rng = np.random.default_rng(stable_seed(seed, "fit", background))
        positions = np.sort(rng.choice(len(part), size=n, replace=False))
        parts.append(part.iloc[positions])
    return pd.concat(parts, ignore_index=True)


@dataclass(frozen=True)
class BTIScaleFit:
    intercept: float
    coefficients: tuple[float, ...]
    feature_names: tuple[str, ...]
    objective: float
    huber_component: float
    tail_component: float
    l1_component: float
    optimizer_success: bool
    optimizer_message: str
    tail_penalty: float
    l1_penalty: float
    tail_quantile: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["coefficient_map"] = dict(zip(self.feature_names, self.coefficients))
        return result


def _validate_frame(frame: pd.DataFrame, feature_names: Sequence[str]) -> None:
    required = {"background_id", "absolute_error", *feature_names}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing BTI-HCP fields: {sorted(missing)}")
    numeric = frame[["absolute_error", *feature_names]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("BTI-HCP inputs contain non-finite values")
    if (frame.absolute_error.to_numpy(float) < 0).any():
        raise ValueError("absolute_error must be nonnegative")


def deployable_scale(
    frame: pd.DataFrame,
    coefficients: Sequence[float],
    feature_names: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Compute the deployable BTI scale; no truth/residual is accessed."""
    x = frame[list(feature_names)].to_numpy(float)
    beta = np.asarray(coefficients, float)
    if beta.shape != (x.shape[1],):
        raise ValueError("Coefficient dimension does not match feature matrix")
    if (beta < 0).any():
        raise ValueError("Monotone BTI coefficients must be nonnegative")
    log_scale = x @ beta
    scale = np.exp(np.clip(log_scale, -30.0, 30.0))
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError("BTI scale is non-finite or non-positive")
    return scale


def _objective_components(
    theta: np.ndarray,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    residual_epsilon: float,
    huber_delta: float,
    tail_quantile: float,
    tail_penalty: float,
    l1_penalty: float,
) -> tuple[float, float, float, float]:
    x = frame[list(feature_names)].to_numpy(float)
    y = np.log(frame.absolute_error.to_numpy(float) + float(residual_epsilon))
    prediction = float(theta[0]) + x @ theta[1:]
    residual = y - prediction
    absolute = np.abs(residual)
    huber = np.where(
        absolute <= huber_delta,
        0.5 * residual**2,
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    weights = background_equal_weights(frame.background_id)
    huber_component = float(np.sum(weights * huber) / np.sum(weights))

    log_scores = y - prediction
    tail_values = []
    ids = frame.background_id.to_numpy()
    for background in sorted(pd.unique(ids)):
        tail_values.append(float(np.quantile(log_scores[ids == background], tail_quantile)))
    # Log-tail variance is invariant to the intercept/global multiplicative scale.
    tail_component = float(np.var(np.asarray(tail_values, float), ddof=0))
    l1_component = float(np.sum(theta[1:]))
    total = huber_component + tail_penalty * tail_component + l1_penalty * l1_component
    return float(total), huber_component, tail_component, l1_component


def fit_bti_scale(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str] = FEATURE_COLUMNS,
    residual_epsilon: float = 1e-3,
    huber_delta: float = 1.0,
    tail_quantile: float = 0.90,
    tail_penalty: float = 1.0,
    l1_penalty: float = 0.002,
    coefficient_bounds: tuple[float, float] = (0.0, 2.0),
    maxiter: int = 300,
) -> BTIScaleFit:
    """Fit a monotone BTI scale with equal total weight per background."""
    _validate_frame(frame, feature_names)
    lower, upper = map(float, coefficient_bounds)
    if lower < 0 or not lower < upper:
        raise ValueError("Invalid nonnegative coefficient bounds")
    initial = np.r_[
        float(np.median(np.log(frame.absolute_error.to_numpy(float) + residual_epsilon))),
        np.full(len(feature_names), 0.25, dtype=float),
    ]

    def objective(theta: np.ndarray) -> float:
        return _objective_components(
            theta, frame, feature_names, residual_epsilon, huber_delta,
            tail_quantile, tail_penalty, l1_penalty,
        )[0]

    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=[(-20.0, 20.0)] + [(lower, upper)] * len(feature_names),
        options={"maxiter": int(maxiter), "xtol": 1e-5, "ftol": 1e-7},
    )
    total, huber, tail, l1 = _objective_components(
        result.x, frame, feature_names, residual_epsilon, huber_delta,
        tail_quantile, tail_penalty, l1_penalty,
    )
    return BTIScaleFit(
        intercept=float(result.x[0]),
        coefficients=tuple(float(value) for value in result.x[1:]),
        feature_names=tuple(feature_names),
        objective=total,
        huber_component=huber,
        tail_component=tail,
        l1_component=l1,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        tail_penalty=float(tail_penalty),
        l1_penalty=float(l1_penalty),
        tail_quantile=float(tail_quantile),
    )


def baseline_scale(frame: pd.DataFrame, method: str, mean_power_b: float = 0.89187068) -> np.ndarray:
    if method == "N0_CONSTANT":
        return np.ones(len(frame), dtype=float)
    if method == "N1_MEAN_POWER":
        return (np.maximum(frame.mu_fixed.to_numpy(float), 0.0) + 1.0) ** float(mean_power_b)
    raise ValueError(f"Unknown baseline: {method}")


def interval_metrics(frame: pd.DataFrame, scale: np.ndarray, qhat: float, nominal: float) -> dict[str, float]:
    mu = frame.mu_fixed.to_numpy(float)
    truth = frame.true_emission.to_numpy(float)
    half = float(qhat) * np.asarray(scale, float)
    lower = np.maximum(0.0, mu - half)
    upper = mu + half
    width = upper - lower
    covered = (truth >= lower) & (truth <= upper)
    normalized = width / (np.maximum(mu, 0.0) + 1.0)
    picp = float(covered.mean())
    return {
        "picp": picp,
        "coverage_error": abs(picp - float(nominal)),
        "mpiw": float(width.mean()),
        "median_width": float(np.median(width)),
        "p90_width": float(np.quantile(width, 0.90)),
        "median_normalized_width": float(np.median(normalized)),
        "lower_zero_fraction": float(np.mean(lower == 0.0)),
    }
