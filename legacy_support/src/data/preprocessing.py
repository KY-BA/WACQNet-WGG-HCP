"""Leakage-safe spatial preprocessing and train-only standardizers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def masked_gaussian_lowpass(
    field: np.ndarray,
    valid_mask: np.ndarray,
    sigma_px: float,
    min_weight: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate background by normalized convolution without zero-fill bias."""
    x = np.asarray(field, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(x)
    if x.ndim != 2 or mask.shape != x.shape:
        raise ValueError("field and valid_mask must be same-shape 2-D arrays")
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    filled = np.where(mask, x, 0.0)
    weights = gaussian_filter(mask.astype(float), sigma=sigma_px, mode="constant", cval=0.0)
    numerator = gaussian_filter(filled, sigma=sigma_px, mode="constant", cval=0.0)
    background = np.full_like(x, np.nan)
    reliable = mask & (weights >= min_weight)
    background[reliable] = numerator[reliable] / weights[reliable]
    return background.astype(np.float32), weights.astype(np.float32)


def build_residual(
    xco2: np.ndarray, valid_mask: np.ndarray, sigma_px: float, min_weight: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Return XCO2 minus a masked normalized-Gaussian background estimate."""
    background, weights = masked_gaussian_lowpass(xco2, valid_mask, sigma_px, min_weight)
    residual = np.where(np.asarray(valid_mask, bool), xco2 - background, np.nan)
    return residual.astype(np.float32), weights


def distance_source_to_edge_km(
    source_yx: tuple[float, float], shape: tuple[int, int], resolution_km: float
) -> float:
    """Distance from source pixel center to the closest grid boundary."""
    y, x = source_yx
    h, w = shape
    return float(max(0.0, min(y + 0.5, x + 0.5, h - y - 0.5, w - x - 0.5)) * resolution_km)


@dataclass
class Standardizer:
    """NaN-aware feature standardizer fitted only on the training partition."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, x: np.ndarray) -> "Standardizer":
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("Training standardizer input must be non-empty [N,D]")
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        if not np.all(np.isfinite(self.mean)):
            raise ValueError("At least one quality feature is entirely missing in training data")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer must be fitted on Train before transform")
        return ((np.asarray(x) - self.mean) / self.std).astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save an unfitted standardizer")
        Path(path).write_text(
            json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist(), "fit_partition": "train"}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.asarray(payload["mean"]), np.asarray(payload["std"]))

