"""Non-oracle wind-conditioned Operational Matched Filter."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TemplateConfig:
    sigma0_km: float = 2.0
    lateral_growth_rate: float = 0.12
    decay_length_km: float = 30.0
    max_downwind_distance_km: float = 60.0
    source_exclusion_km: float = 1.0
    min_wind_speed: float = 0.2
    v_positive_toward_increasing_row: bool = True


def wind_components_to_direction(u: float, v: float) -> tuple[float, float, float]:
    """Return unit downwind vector and speed from eastward/northward components."""
    speed = float(np.hypot(u, v))
    if not np.isfinite(speed):
        raise ValueError("Wind components must be finite")
    denom = max(speed, 1e-12)
    return float(u / denom), float(v / denom), speed


def rotated_coordinates(
    shape: tuple[int, int],
    source_yx: tuple[float, float],
    u: float,
    v: float,
    resolution_km: float,
    v_positive_toward_increasing_row: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return along/cross-wind coordinates under an explicit array convention.

    SMARTCARB arrays used by the legacy TensorFlow experiment encode positive
    ``v`` toward increasing row indices.  Other products can opt into the
    geographic-image convention (positive ``v`` toward decreasing rows).
    """
    h, w = shape
    yy, xx = np.indices((h, w), dtype=float)
    dx = (xx - source_yx[1]) * resolution_km
    row_distance = (yy - source_yx[0]) * resolution_km
    dy_v = row_distance if v_positive_toward_increasing_row else -row_distance
    ux, uy, _ = wind_components_to_direction(u, v)
    parallel = dx * ux + dy_v * uy
    cross = -dx * uy + dy_v * ux
    return parallel, cross


def generate_operational_template(
    shape: tuple[int, int],
    source_yx: tuple[float, float],
    u: float,
    v: float,
    resolution_km: float,
    valid_mask: np.ndarray,
    config: TemplateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a wind-only Gaussian cone; no injected plume input is accepted."""
    valid = np.asarray(valid_mask, bool)
    if valid.shape != shape:
        raise ValueError("valid_mask shape differs from requested template shape")
    parallel, cross = rotated_coordinates(
        shape, source_yx, u, v, resolution_km, config.v_positive_toward_increasing_row
    )
    _, _, speed = wind_components_to_direction(u, v)
    downwind = (parallel > config.source_exclusion_km) & (parallel <= config.max_downwind_distance_km)
    sigma_y = config.sigma0_km + config.lateral_growth_rate * np.maximum(parallel, 0.0)
    raw = np.exp(-np.maximum(parallel, 0.0) / config.decay_length_km)
    raw *= np.exp(-(cross**2) / (2.0 * np.maximum(sigma_y, 1e-6) ** 2))
    support = valid & downwind & (speed >= config.min_wind_speed)
    template = np.where(support, raw, 0.0)
    if np.count_nonzero(support) < 2:
        return np.zeros(shape, np.float32), support
    # Project out the constant background only within plume support.
    template[support] -= template[support].mean()
    norm = np.linalg.norm(template[support])
    if norm <= 1e-12:
        return np.zeros(shape, np.float32), support
    template /= norm
    return template.astype(np.float32), support


def make_background_reference_mask(
    shape: tuple[int, int],
    source_yx: tuple[float, float],
    u: float,
    v: float,
    resolution_km: float,
    valid_mask: np.ndarray,
    config: TemplateConfig,
    plume_exclusion_sigma: float = 3.0,
    source_buffer_km: float = 4.0,
) -> np.ndarray:
    """Prefer upwind/far-crosswind valid pixels and exclude source/downwind cone."""
    parallel, cross = rotated_coordinates(
        shape, source_yx, u, v, resolution_km, config.v_positive_toward_increasing_row
    )
    sigma_y = config.sigma0_km + config.lateral_growth_rate * np.maximum(parallel, 0.0)
    contaminated = (
        (parallel > 0)
        & (parallel <= config.max_downwind_distance_km)
        & (np.abs(cross) <= plume_exclusion_sigma * sigma_y)
    )
    radial = np.hypot(parallel, cross)
    return np.asarray(valid_mask, bool) & ~contaminated & (radial >= source_buffer_km)


def robust_background_sigma(values: np.ndarray, method: str = "mad", winsor_quantile: float = 0.05) -> float:
    """Estimate scalar background noise from unpolluted residual pixels."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    if method == "mad":
        sigma = 1.4826 * np.median(np.abs(x - np.median(x)))
    elif method == "winsorized":
        lo, hi = np.quantile(x, [winsor_quantile, 1.0 - winsor_quantile])
        sigma = np.std(np.clip(x, lo, hi), ddof=1)
    elif method == "sample":
        sigma = np.std(x, ddof=1)
    else:
        raise ValueError(f"Unknown variance estimator: {method}")
    return float(max(sigma, 1e-8))


def matched_filter_score(
    residual: np.ndarray,
    template: np.ndarray,
    background_mask: np.ndarray,
    variance_method: str = "mad",
    min_background_pixels: int = 100,
) -> tuple[float, dict[str, float | int | bool]]:
    """Compute p^T Sigma^-1 x / sqrt(p^T Sigma^-1 p) for scalar Sigma."""
    x = np.asarray(residual, float)
    p = np.asarray(template, float)
    valid = np.isfinite(x) & np.isfinite(p) & (p != 0)
    n_bg = int(np.count_nonzero(background_mask & np.isfinite(x)))
    sigma = robust_background_sigma(x[np.asarray(background_mask, bool)], variance_method)
    ok = n_bg >= min_background_pixels and np.isfinite(sigma) and np.count_nonzero(valid) >= 2
    if not ok:
        return float("nan"), {
            "background_pixels": n_bg,
            "sigma_background": sigma,
            "quality_ok": False,
            "quality_flag": "insufficient_background_pixels_or_template_support",
        }
    inv_var = 1.0 / (sigma * sigma)
    numerator = float(np.sum(p[valid] * x[valid] * inv_var))
    denominator = float(np.sqrt(np.sum(p[valid] ** 2 * inv_var)))
    score = numerator / max(denominator, 1e-12)
    return score, {
        "background_pixels": n_bg,
        "sigma_background": sigma,
        "quality_ok": True,
        "quality_flag": "ok",
    }


def fit_false_alarm_threshold(class0_scores: Iterable[float], false_alarm_rate: float = 0.05) -> float:
    """Fit tau0 from Train/Validation Class-0 scores only."""
    if not 0 < false_alarm_rate < 1:
        raise ValueError("false_alarm_rate must lie in (0,1)")
    values = np.asarray(list(class0_scores), float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite Class-0 Train/Validation scores available")
    return float(np.quantile(values, 1.0 - false_alarm_rate, method="higher"))


def detection_probability(scores: Iterable[float], tau0: float) -> float:
    """Empirical detection probability across repeated perturbations."""
    values = np.asarray(list(scores), float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values > tau0))


def resampled_detection_scores(
    residual: np.ndarray,
    template: np.ndarray,
    background_mask: np.ndarray,
    repetitions: int,
    seed: int,
    variance_method: str = "mad",
    min_background_pixels: int = 30,
) -> np.ndarray:
    """Generate repeated operational scores by same-scene background resampling.

    The observed target residual is held fixed. A zero-mean background-noise
    realization is sampled with replacement from the unpolluted reference
    pixels and added only on template support. This estimates detectability
    under repeated noise without ever accepting an oracle plume template.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    x = np.asarray(residual, float)
    p = np.asarray(template, float)
    background_values = x[np.asarray(background_mask, bool) & np.isfinite(x)]
    support = np.isfinite(x) & (p != 0)
    if background_values.size < min_background_pixels or np.count_nonzero(support) < 2:
        return np.full(repetitions, np.nan)
    background_values = background_values - np.median(background_values)
    rng = np.random.default_rng(seed)
    scores = np.empty(repetitions, float)
    for repetition in range(repetitions):
        perturbed = x.copy()
        perturbed[support] += rng.choice(background_values, size=np.count_nonzero(support), replace=True)
        scores[repetition], _ = matched_filter_score(
            perturbed, p, background_mask, variance_method, min_background_pixels
        )
    return scores


def hard_class(emission: float, p_det: float, quantifiable_probability_threshold: float = 0.8) -> int:
    """Independent matched-filter label: background, sub-detection, quantifiable."""
    if emission <= 0:
        return 0
    if not np.isfinite(p_det):
        raise ValueError("Positive-emission samples require a finite p_det")
    return 1 if p_det < quantifiable_probability_threshold else 2
