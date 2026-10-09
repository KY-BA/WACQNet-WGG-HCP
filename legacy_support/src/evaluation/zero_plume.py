"""Auditable utilities for plume-free OCO-3 forced-regression diagnostics.

The XCO2 channel is read only from each original ``*_scene.npz`` file's
``xco2_grid`` field.  This module never obtains Class 0 by relabelling a mixed
sample or subtracting a plume from a mixed sample.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.special import betainc


@dataclass(frozen=True)
class ZeroPlumeScene:
    """One original, independently gridded real OCO-3 background scene."""

    background_id: str
    scene_id: str
    background_type: str
    library: str
    split: str
    path: Path
    xco2: np.ndarray
    valid_mask: np.ndarray
    residual: np.ndarray
    background_low: np.ndarray
    metadata: dict[str, Any]


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for provenance records."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_zero_plume_scenes(
    manifest_path: str | Path,
    legacy_repo: str | Path,
    *,
    direct_xco2_field: str = "xco2_grid",
    residual_field: str = "background_residual",
    lowpass_field: str = "background_low",
    expected_count: int | None = 13,
) -> list[ZeroPlumeScene]:
    """Load direct pure backgrounds named by the second-paper manifest.

    The proof obligation is structural: the selected NPZ must contain a direct
    background grid and the output is copied from that field.  Neither a
    ``mixed`` nor a ``plume`` array is read to construct XCO2.
    """
    manifest = Path(manifest_path)
    root = Path(legacy_repo)
    if not manifest.exists():
        raise FileNotFoundError(f"Background manifest not found: {manifest}")
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} backgrounds, found {len(rows)}")
    scenes: list[ZeroPlumeScene] = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Manifest background file is missing: {path}")
        with np.load(path, allow_pickle=False) as payload:
            keys = tuple(payload.files)
            required = (direct_xco2_field, residual_field, lowpass_field)
            missing = [key for key in required if key not in payload]
            if missing:
                raise KeyError(f"Direct background file {path} lacks fields: {missing}")
            xco2 = np.asarray(payload[direct_xco2_field], dtype=np.float32)
            residual = np.asarray(payload[residual_field], dtype=np.float32)
            low = np.asarray(payload[lowpass_field], dtype=np.float32)
            if xco2.ndim != 2 or residual.shape != xco2.shape or low.shape != xco2.shape:
                raise ValueError(f"Invalid background grid shapes in {path}")
            valid = np.isfinite(xco2)
            if not np.any(valid):
                raise ValueError(f"Background contains no valid XCO2 pixels: {path}")
            # Stored residual must independently close the direct-background
            # identity; this is an audit check, not a reconstruction step.
            closure = np.nanmax(np.abs((low + residual - xco2)[valid]))
            if not np.isfinite(closure) or closure > 2e-4:
                raise ValueError(f"Stored low+residual identity failed for {path}: {closure}")
            center_lat = float(payload["center_lat"]) if "center_lat" in payload else np.nan
            center_lon = float(payload["center_lon"]) if "center_lon" in payload else np.nan
            neighbor_count = np.asarray(payload["neighbor_count"]) if "neighbor_count" in payload else None
        scene_id = row.get("background_name") or path.stem.replace("_scene", "")
        scenes.append(
            ZeroPlumeScene(
                background_id=scene_id,
                scene_id=scene_id,
                background_type=row.get("category", "unknown"),
                library=row.get("library", "unknown"),
                split=row.get("split", "unknown"),
                path=path.resolve(),
                xco2=xco2,
                valid_mask=valid,
                residual=residual,
                background_low=low,
                metadata={
                    "npz_fields": list(keys),
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "valid_pixels": int(valid.sum()),
                    "valid_fraction": float(valid.mean()),
                    "neighbor_count_min": int(np.min(neighbor_count)) if neighbor_count is not None else None,
                    "neighbor_count_max": int(np.max(neighbor_count)) if neighbor_count is not None else None,
                    "residual_identity_max_abs_error": float(closure),
                    "class0_source": f"direct_npz_field:{direct_xco2_field}",
                    "target_plume_added": False,
                },
            )
        )
    ids = [scene.background_id for scene in scenes]
    if len(ids) != len(set(ids)):
        raise ValueError("background_id values in the manifest are not unique")
    return scenes


def load_fixed_legacy_wind(
    dataset_path: str | Path, row_index: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Read one pre-specified u/v field from the exact second-paper source library."""
    # The NetCDF file is HDF5-backed; direct h5py reads avoid loading unrelated
    # plume/XCO2 fields and work in both the audit and TensorFlow environments.
    with h5py.File(dataset_path, "r") as dataset:
        n = int(dataset["u"].shape[0])
        if not 0 <= row_index < n:
            raise IndexError(f"fixed wind row {row_index} outside [0,{n})")
        u = np.asarray(dataset["u"][row_index], dtype=np.float32)
        v = np.asarray(dataset["v"][row_index], dtype=np.float32)
    if u.shape != v.shape or u.ndim != 2 or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("Selected legacy u/v fields are invalid")
    wind_u = float(np.mean(u))
    wind_v = float(np.mean(v))
    return u, v, {
        "wind_row_index": int(row_index),
        "wind_u": wind_u,
        "wind_v": wind_v,
        "wind_speed": float(np.hypot(wind_u, wind_v)),
    }


def compose_zero_plume_inputs(
    scenes: Iterable[ZeroPlumeScene], u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Stack direct background XCO2 with fixed legacy u/v in historical order."""
    scene_list = list(scenes)
    if not scene_list:
        raise ValueError("At least one zero-plume scene is required")
    shape = scene_list[0].xco2.shape
    if u.shape != shape or v.shape != shape:
        raise ValueError(f"Wind shape {u.shape}/{v.shape} does not match background {shape}")
    images = np.empty((len(scene_list), *shape, 3), dtype=np.float32)
    masks = np.empty((len(scene_list), *shape), dtype=bool)
    for index, scene in enumerate(scene_list):
        if scene.xco2.shape != shape:
            raise ValueError("All background scenes must share one grid shape")
        images[index, ..., 0] = scene.xco2
        images[index, ..., 1] = u
        images[index, ..., 2] = v
        masks[index] = scene.valid_mask
    return images, masks


def compute_legacy_nan_fill_values(train_h5: str | Path, field_name: str) -> np.ndarray:
    """Reproduce the old Input_train_mixed exact Train-only channel medians.

    The second-paper pipeline did not persist these three values.  Therefore
    they are deterministically recomputed from its original Train HDF5, never
    from the 13 diagnostic backgrounds.
    """
    with h5py.File(train_h5, "r") as handle:
        if field_name not in handle:
            raise KeyError(f"Legacy training field not found: {field_name}")
        dataset = handle[field_name]
        if dataset.ndim == 3:
            array = np.asarray(dataset[:], dtype=np.float32)
            values = np.array([np.nanmedian(array)], dtype=np.float32)
        elif dataset.ndim == 4:
            # Exact, but channel-wise to avoid materialising the compressed
            # ~2 GB corpus and all temporary median workspaces simultaneously.
            medians = []
            for channel in range(dataset.shape[-1]):
                array = np.asarray(dataset[..., channel], dtype=np.float32)
                medians.append(float(np.nanmedian(array)))
                del array
            values = np.asarray(medians, dtype=np.float32)
        else:
            raise ValueError(f"Legacy training field must be [N,H,W] or [N,H,W,C], got {dataset.shape}")
    values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    return values


def fill_per_channel(images: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    """Apply the exact historical per-channel missing-value rule."""
    output = np.array(images, copy=True, dtype=np.float32)
    fill = np.asarray(fill_values, dtype=np.float32).reshape(-1)
    if output.ndim != 4 or fill.size != output.shape[-1]:
        raise ValueError("Expected images [N,H,W,C] and one fill value per channel")
    for channel in range(output.shape[-1]):
        output[..., channel] = np.where(
            np.isfinite(output[..., channel]), output[..., channel], fill[channel]
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Non-finite values remain after legacy missing-value filling")
    return output


def build_prediction_frame(
    scenes: list[ZeroPlumeScene],
    predictions: np.ndarray,
    wind: dict[str, float | int],
    *,
    model_name: str,
    seed: int,
    checkpoint: str | Path,
    source_x: float,
    source_y: float,
) -> pd.DataFrame:
    """Create the requested per-background diagnostic table."""
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    if pred.size != len(scenes) or not np.all(np.isfinite(pred)):
        raise ValueError("Real model predictions must be finite and one per scene")
    rows: list[dict[str, Any]] = []
    for scene, value in zip(scenes, pred):
        valid_x = scene.xco2[scene.valid_mask]
        valid_r = scene.residual[scene.valid_mask & np.isfinite(scene.residual)]
        center = float(np.median(valid_r))
        rows.append(
            {
                "background_id": scene.background_id,
                "scene_id": scene.scene_id,
                "background_type": scene.background_type,
                "legacy_split": scene.split,
                "background_library": scene.library,
                "source_x": source_x,
                "source_y": source_y,
                "predicted_emission_mt_yr": float(value),
                "predicted_emission_clipped_mt_yr": float(max(0.0, value)),
                "xco2_mean": float(np.mean(valid_x)),
                "xco2_std": float(np.std(valid_x)),
                "residual_std": float(np.std(valid_r)),
                "residual_mad_sigma": float(1.4826 * np.median(np.abs(valid_r - center))),
                "valid_fraction": float(scene.valid_mask.mean()),
                "wind_u": wind["wind_u"],
                "wind_v": wind["wind_v"],
                "wind_speed": wind["wind_speed"],
                "wind_row_index": wind["wind_row_index"],
                "model_name": model_name,
                "seed": int(seed),
                "checkpoint": str(Path(checkpoint).resolve()),
                "class0_source": "direct_original_npz_xco2_grid",
                "target_plume_added": False,
            }
        )
    return pd.DataFrame(rows)


def prediction_statistics(values: np.ndarray) -> dict[str, float | int]:
    """Summarize raw, unclipped forced-regression output."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Prediction statistics require non-empty finite values")
    quantiles = np.quantile(x, [0.25, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "minimum": float(np.min(x)),
        "p25": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "maximum": float(np.max(x)),
        "fraction_gt_1": float(np.mean(x > 1.0)),
        "fraction_gt_2": float(np.mean(x > 2.0)),
        "fraction_gt_5": float(np.mean(x > 5.0)),
        "fraction_gt_10": float(np.mean(x > 10.0)),
        "fraction_lt_0": float(np.mean(x < 0.0)),
    }


def background_type_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return type-level descriptive statistics without treating rows as replicates."""
    rows = []
    for name, group in frame.groupby("background_type", sort=True):
        pred = group["predicted_emission_mt_yr"].to_numpy(float)
        rows.append(
            {
                "background_type": name,
                "n": len(group),
                "mean_predicted_emission_mt_yr": float(np.mean(pred)),
                "median_predicted_emission_mt_yr": float(np.median(pred)),
                "p90_predicted_emission_mt_yr": float(np.quantile(pred, 0.90)),
                "p95_predicted_emission_mt_yr": float(np.quantile(pred, 0.95)),
                "max_predicted_emission_mt_yr": float(np.max(pred)),
                "mean_residual_std": float(group["residual_std"].mean()),
                "mean_valid_fraction": float(group["valid_fraction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def exploratory_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Spearman diagnostics; constant covariates are reported as undefined."""
    rows = []
    y = frame["predicted_emission_mt_yr"].to_numpy(float)
    for column in ("residual_std", "valid_fraction", "wind_speed"):
        x = frame[column].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3 or np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
            rho, pvalue = np.nan, np.nan
            note = "undefined: fewer than 3 finite values or a constant variable"
        else:
            # Average-rank Spearman coefficient without NumPy BLAS calls.  The
            # legacy TensorFlow 2.9 Windows environment has an incompatible
            # SciPy/BLAS path in scipy.stats.spearmanr; the formula below is
            # mathematically equivalent, with the usual asymptotic t p-value.
            x_rank = pd.Series(x[finite]).rank(method="average").to_numpy(float)
            y_rank = pd.Series(y[finite]).rank(method="average").to_numpy(float)
            x_centered = x_rank - float(np.mean(x_rank))
            y_centered = y_rank - float(np.mean(y_rank))
            numerator = float(np.sum(x_centered * y_centered))
            denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
            rho = numerator / denominator
            degrees = int(finite.sum()) - 2
            if abs(rho) >= 1.0:
                pvalue = 0.0
            else:
                t_squared = rho * rho * degrees / max(1e-15, 1.0 - rho * rho)
                pvalue = float(betainc(0.5 * degrees, 0.5, degrees / (degrees + t_squared)))
            note = "exploratory; asymptotic two-sided p-value; no multiplicity correction"
        rows.append({"predictor": column, "spearman_rho": rho, "p_value": pvalue, "n": int(finite.sum()), "note": note})
    return pd.DataFrame(rows)


def classify_hypothesis(stats: dict[str, float | int], rubric: dict[str, float]) -> tuple[str, str]:
    """Apply the pre-specified, configuration-recorded diagnostic rubric."""
    if (
        float(stats["median"]) >= float(rubric["strongly_supported_median_mt_yr"])
        and float(stats["fraction_gt_1"]) >= float(rubric["strongly_supported_fraction_gt_1"])
    ):
        return "A. Strongly supported", "Pure backgrounds show a systematic, practically non-negligible positive forced-regression output."
    if (
        float(stats["p95"]) >= float(rubric["partially_supported_p95_mt_yr"])
        or float(stats["maximum"]) >= float(rubric["partially_supported_max_mt_yr"])
    ):
        return "B. Partially supported", "The overall output is near zero, but a subset of backgrounds produces non-negligible positive estimates."
    return "C. Not supported", "Pure-background predictions are stably near zero with few or no practically large extremes."


def save_json(payload: Any, path: str | Path) -> None:
    """Write JSON while preserving explicit NaN as null."""
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (np.floating, float)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    Path(path).write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")
