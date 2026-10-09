"""Scene-adaptive null controls for an operational matched-filter detector.

The public APIs intentionally accept no true-plume or oracle-mask argument.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.data.matched_filter import (
    TemplateConfig, generate_operational_template, make_background_reference_mask,
    matched_filter_score,
)


@dataclass(frozen=True)
class NullControl:
    """One deterministic pseudo-source template and its background mask."""

    source_y: float
    source_x: float
    template: np.ndarray
    background_mask: np.ndarray


def deterministic_null_controls(
    valid_mask: np.ndarray,
    target_source_yx: tuple[float, float],
    u: float,
    v: float,
    resolution_km: float,
    template_config: TemplateConfig,
    scene_key: str,
    seed: int,
    k_null: int = 32,
    min_source_distance_pixels: float = 6.0,
    max_template_overlap: float = 0.10,
    plume_exclusion_sigma: float = 3.0,
    source_buffer_km: float = 4.0,
    min_background_pixels: int = 100,
) -> tuple[list[NullControl], str]:
    """Pre-register pseudo sources without inspecting their matched-filter scores."""
    valid = np.asarray(valid_mask, bool)
    target, _ = generate_operational_template(
        valid.shape, target_source_yx, u, v, resolution_km, valid, template_config
    )
    candidates: list[tuple[str, int, int]] = []
    yy, xx = np.argwhere(valid).T
    for y, x in zip(yy, xx):
        if np.hypot(float(y) - target_source_yx[0], float(x) - target_source_yx[1]) < min_source_distance_pixels:
            continue
        token = hashlib.sha256(f"{seed}|{scene_key}|{int(y)}|{int(x)}".encode()).hexdigest()
        candidates.append((token, int(y), int(x)))
    controls: list[NullControl] = []
    for _, y, x in sorted(candidates):
        pseudo, support = generate_operational_template(
            valid.shape, (float(y), float(x)), u, v, resolution_km, valid, template_config
        )
        if np.count_nonzero(support) < 2:
            continue
        overlap = abs(float(np.sum(target * pseudo)))
        if overlap >= max_template_overlap:
            continue
        background = make_background_reference_mask(
            valid.shape, (float(y), float(x)), u, v, resolution_km, valid,
            template_config, plume_exclusion_sigma, source_buffer_km,
        )
        if np.count_nonzero(background) < min_background_pixels:
            continue
        controls.append(NullControl(float(y), float(x), pseudo, background))
        if len(controls) == k_null:
            break
    if len(controls) >= k_null:
        flag = "ok"
    elif len(controls) >= 16:
        flag = "reduced_null_controls"
    else:
        flag = "insufficient_null_controls"
    return controls, flag


def scene_normalized_score(
    residual: np.ndarray,
    target_score: float,
    controls: Iterable[NullControl],
    variance_method: str = "mad",
    min_background_pixels: int = 100,
    epsilon: float = 1e-6,
) -> tuple[float, dict[str, float | int | str]]:
    """Robustly standardize a target score by deterministic pseudo-source scores."""
    scores = []
    for control in controls:
        score, quality = matched_filter_score(
            residual, control.template, control.background_mask,
            variance_method, min_background_pixels,
        )
        if quality["quality_ok"] and np.isfinite(score):
            scores.append(float(score))
    values = np.asarray(scores, float)
    if values.size < 16:
        return float("nan"), {"null_score_median": float("nan"), "null_score_mad_sigma": float("nan"),
                              "n_null_controls": int(values.size), "null_quality_flag": "insufficient_valid_null_scores"}
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if not np.isfinite(scale) or scale <= epsilon:
        return float("nan"), {"null_score_median": median, "null_score_mad_sigma": scale,
                              "n_null_controls": int(values.size), "null_quality_flag": "degenerate_null_mad"}
    return float((target_score - median) / (scale + epsilon)), {
        "null_score_median": median, "null_score_mad_sigma": scale,
        "n_null_controls": int(values.size), "null_quality_flag": "ok",
    }
