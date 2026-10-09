"""Frozen Level-B construction shared by new BTI calibration and test stages."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml


def stable_seed(base: int, *parts: object) -> int:
    token = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def _legacy_symbols(legacy_root: str):
    root = str(Path(legacy_root))
    if root not in sys.path:
        sys.path.append(root)
    # ``src`` is already imported from the new auditable project.  Extend its
    # package search path explicitly so the frozen legacy implementation can
    # be used read-only without copying or modifying that project.
    import src as source_package
    legacy_source = str(Path(root) / "src")
    if legacy_source not in source_package.__path__:
        source_package.__path__.append(legacy_source)
    from scripts.build_reliability_boundary import FrozenHPR
    from src.data.detectability_experiment import select_physical_rows, spatial_block_bootstrap
    from src.data.matched_filter import TemplateConfig, generate_operational_template
    from src.evaluation.reliability_boundary import block_correlated_mask, operational_downwind_mask, rotate_wind
    return FrozenHPR, select_physical_rows, spatial_block_bootstrap, TemplateConfig, generate_operational_template, block_correlated_mask, operational_downwind_mask, rotate_wind


def _median_wind(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float, float]:
    uu, vv = float(np.nanmedian(u)), float(np.nanmedian(v))
    return uu, vv, float(np.hypot(uu, vv)), float(np.degrees(np.arctan2(vv, uu)))


def generate_levelb(
    cfg: dict[str, Any],
    manifest: pd.DataFrame,
    *,
    role: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate frozen HPR predictions for one explicitly authorized role."""
    symbols = _legacy_symbols(cfg["legacy_project_read_only"])
    FrozenHPR, select_physical_rows, spatial_block_bootstrap, TemplateConfig, generate_operational_template, block_correlated_mask, operational_downwind_mask, rotate_wind = symbols
    level = cfg["level_b"]
    with h5py.File(level["plume_dataset"], "r") as source:
        plume = np.asarray(source["plume"][:], np.float32)
        u_all = np.asarray(source["u"][:], np.float32)
        v_all = np.asarray(source["v"][:], np.float32)
        emission_ref = np.asarray(source["emiss"][:, 1], np.float32)
    physical_rows = select_physical_rows(u_all, v_all, list(level["physical_wind_quantiles"]))
    if physical_rows != list(level["expected_physical_rows"]):
        raise RuntimeError(f"Frozen physical-row selection drift: {physical_rows}")
    detector = yaml.safe_load(Path(cfg["operational_template"]["detector_config"]).read_text(encoding="utf-8"))["detector"]
    template_cfg = TemplateConfig(**detector["template"])
    source_yx = (float(detector["source_y"]), float(detector["source_x"]))
    hpr_cfg = {"backbone": cfg["backbone"], "input": {"legacy_nan_fill": cfg["backbone"]["nan_fill"]}}
    hpr = FrozenHPR(hpr_cfg, output_dir)

    emissions = [float(x) for x in level["emission_grid_mt_yr"]]
    core = {float(x) for x in level["core_emissions_mt_yr"]}
    base_row = int(level["base_physical_wind_row"])
    repetitions = int(level["repetitions"])
    batch_size = int(level["batch_size"])
    expected_per_background = int(level["expected_rows_per_background"])
    expected_total = len(manifest) * expected_per_background
    project_root = Path(cfg["project_root"])
    records: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    pending: list[dict[str, Any]] = []
    hpr_calls = 0

    def flush() -> None:
        nonlocal images, pending, hpr_calls
        if not images:
            return
        prediction = hpr.predict(images, batch_size=batch_size)
        hpr_calls += 1
        for row, mu in zip(pending, prediction):
            row["mu_fixed"] = float(mu)
            records.append(row)
        print(f"{role} frozen HPR {len(records)}/{expected_total}", flush=True)
        images, pending = [], []

    def emit(image: np.ndarray, valid: np.ndarray, metadata: dict[str, Any]) -> None:
        metadata["valid_fraction"] = float(valid.mean())
        images.append(image.astype(np.float32))
        pending.append(metadata)
        if len(images) >= batch_size:
            flush()

    for bg in manifest.sort_values("background_id").itertuples(index=False):
        scene_path = Path(bg.scene_file)
        if not scene_path.is_absolute():
            scene_path = project_root / scene_path
        with np.load(scene_path) as scene:
            low = np.asarray(scene["background_low"], np.float32)
            residual_base = np.asarray(scene["background_residual"], np.float32)
            valid = np.asarray(scene["valid_mask"], bool)
        base_images: dict[tuple[float, int], np.ndarray] = {}
        for wind_row in physical_rows:
            true_u, true_v = u_all[wind_row], v_all[wind_row]
            _, _, wind_speed, wind_direction = _median_wind(true_u, true_v)
            for emission in emissions:
                condition = f"{bg.background_id}|windrow={wind_row}|Q={emission:g}"
                scaling = emission / float(emission_ref[wind_row])
                for repetition in range(repetitions):
                    rng = np.random.default_rng(stable_seed(int(cfg["seed"]), bg.background_id, wind_row, emission, repetition))
                    noise = spatial_block_bootstrap(residual_base, int(level["spatial_block_size_px"]), rng)
                    xco2 = np.where(valid, low + noise + plume[wind_row] * scaling, np.nan).astype(np.float32)
                    image = np.stack((xco2, true_u, true_v), axis=-1).astype(np.float32)
                    if wind_row == base_row and emission in core:
                        base_images[(emission, repetition)] = image.copy()
                    emit(image, valid, {
                        "sample_id": f"{condition}|r={repetition:03d}", "condition_id": condition,
                        "background_id": bg.background_id, "background_group_id": bg.background_group_id,
                        "background_type": bg.background_type, "role": role, "experiment": "physical_wind",
                        "true_emission": emission, "true_wind_speed": wind_speed,
                        "true_wind_direction": wind_direction, "retained_valid_fraction": 1.0,
                        "missingness_type": "natural_oco3_mask", "source_library_row": wind_row,
                        "repetition": repetition,
                    })
        for emission in sorted(core):
            true_u, true_v = u_all[base_row], v_all[base_row]
            _, _, wind_speed, wind_direction = _median_wind(true_u, true_v)
            for repetition in range(repetitions):
                base_image = base_images[(emission, repetition)]
                interventions: list[tuple[str, float, str]] = []
                interventions += [("wind_direction_error", float(x), "none") for x in level["wind_direction_errors_deg"] if float(x) != 0.0]
                interventions += [("wind_speed_input_error", float(x), "none") for x in level["wind_speed_multipliers"] if float(x) != 1.0]
                interventions += [("missingness", float(retained), str(kind)) for kind in level["missingness_types"] for retained in level["retained_valid_fractions"] if float(retained) != 1.0]
                for experiment, parameter, missingness in interventions:
                    image = base_image.copy()
                    new_valid = valid.copy()
                    if experiment == "wind_direction_error":
                        image[..., 1], image[..., 2] = rotate_wind(true_u, true_v, parameter, 1.0)
                    elif experiment == "wind_speed_input_error":
                        image[..., 1], image[..., 2] = rotate_wind(true_u, true_v, 0.0, parameter)
                    else:
                        key = f"{bg.background_id}|Q={emission:g}|r={repetition}|{missingness}|{parameter:g}"
                        if missingness == "block_correlated":
                            new_valid = block_correlated_mask(valid, parameter, int(level["missingness_block_size_px"]), key, int(cfg["seed"]))
                        else:
                            template, _ = generate_operational_template(valid.shape, source_yx, float(np.nanmedian(true_u)), float(np.nanmedian(true_v)), float(detector["grid_resolution_km"]), valid, template_cfg)
                            new_valid = operational_downwind_mask(valid, template, parameter)
                        image[..., 0] = np.where(new_valid, image[..., 0], np.nan)
                    condition = f"{bg.background_id}|Q={emission:g}|windrow={base_row}|{experiment}|parameter={parameter:g}|missing={missingness}"
                    emit(image, new_valid, {
                        "sample_id": f"{condition}|r={repetition:03d}", "condition_id": condition,
                        "background_id": bg.background_id, "background_group_id": bg.background_group_id,
                        "background_type": bg.background_type, "role": role, "experiment": experiment,
                        "true_emission": emission, "true_wind_speed": wind_speed,
                        "true_wind_direction": wind_direction,
                        "retained_valid_fraction": parameter if experiment == "missingness" else 1.0,
                        "missingness_type": missingness, "source_library_row": base_row,
                        "repetition": repetition,
                    })
        print(f"completed {role}: {bg.background_id}", flush=True)
    flush()
    result = pd.DataFrame(records)
    if len(result) != expected_total or result.background_id.nunique() != len(manifest):
        raise RuntimeError(f"Level-B count mismatch: {len(result)} rows, {result.background_id.nunique()} groups")
    counts = result.groupby("background_id").size()
    conditions = result.groupby("background_id").condition_id.nunique()
    if not (counts == expected_per_background).all():
        raise RuntimeError(f"Per-background realization mismatch: {counts.to_dict()}")
    if not (conditions == int(level["expected_conditions_per_background"])).all():
        raise RuntimeError(f"Per-background condition mismatch: {conditions.to_dict()}")
    audit = {
        "role": role, "backgrounds_read": int(result.background_id.nunique()),
        "rows_read": int(len(result)), "hpr_calls": int(hpr_calls), "v2_calls": 0,
        "rows_per_background": {str(k): int(v) for k, v in counts.items()},
        "conditions_per_background": {str(k): int(v) for k, v in conditions.items()},
        "q_range": [float(result.true_emission.min()), float(result.true_emission.max())],
        "wind_range": [float(result.true_wind_speed.min()), float(result.true_wind_speed.max())],
        "valid_fraction_range": [float(result.valid_fraction.min()), float(result.valid_fraction.max())],
    }
    return result, audit
