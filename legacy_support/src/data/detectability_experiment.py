"""Formal, leakage-safe construction of Operational Matched-Filter labels.

True SMARTCARB plume arrays are used only as known synthetic injections and in
a separate Train-only coordinate audit. They are never accepted by the
operational-template or score APIs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import yaml

from src.data.matched_filter import (
    TemplateConfig,
    detection_probability,
    fit_false_alarm_threshold,
    generate_operational_template,
    hard_class,
    make_background_reference_mask,
    matched_filter_score,
)
from src.data.preprocessing import build_residual, distance_source_to_edge_km


def validate_frozen_groups(split_config: dict[str, list[str]], available: Iterable[str]) -> dict[str, str]:
    """Return background-to-partition mapping after strict disjointness checks."""
    required = {"train", "validation", "calibration", "test"}
    if set(split_config) != required:
        raise ValueError(f"Split must contain exactly {sorted(required)}")
    owner: dict[str, str] = {}
    for partition, groups in split_config.items():
        for group in groups:
            if group in owner:
                raise ValueError(f"Group leakage: {group} is in {owner[group]} and {partition}")
            owner[group] = partition
    missing = set(available) - set(owner)
    extra = set(owner) - set(available)
    if missing or extra:
        raise ValueError(f"Frozen split mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return owner


def spatial_block_bootstrap(residual: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap that retains within-block spatial correlation."""
    x = np.asarray(residual, float)
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("Cannot resample an all-missing residual")
    centred = np.where(finite, x - np.nanmedian(x), 0.0)
    h, w = centred.shape
    if block_size < 1 or block_size > min(h, w):
        raise ValueError("Invalid spatial block size")
    output = np.empty_like(centred)
    for y in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            bh, bw = min(block_size, h - y), min(block_size, w - x0)
            sy = int(rng.integers(0, h)); sx = int(rng.integers(0, w))
            iy = (np.arange(sy, sy + bh) % h)[:, None]
            ix = (np.arange(sx, sx + bw) % w)[None, :]
            output[y:y + bh, x0:x0 + bw] = centred[iy, ix]
    return output.astype(np.float32)


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values])


def _median_wind(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    uu, vv = float(np.nanmedian(u)), float(np.nanmedian(v))
    return uu, vv, float(np.hypot(uu, vv))


def select_physical_rows(u: np.ndarray, v: np.ndarray, quantiles: list[float]) -> list[int]:
    """Pre-register physical cases by wind-speed quantiles, never by detector score."""
    speed = np.hypot(np.nanmedian(u, axis=(1, 2)), np.nanmedian(v, axis=(1, 2)))
    order = np.argsort(speed, kind="stable")
    selected = [int(order[min(len(order) - 1, round(q * (len(order) - 1)))]) for q in quantiles]
    if len(set(selected)) != len(selected):
        raise ValueError("Wind quantiles selected duplicate physical rows")
    return selected


def coordinate_feasibility_audit(
    plume: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    rows: list[int],
    source_yx: tuple[float, float],
    template_config: TemplateConfig,
) -> dict[str, Any]:
    """Check Train-only array/wind orientation without using plume in scoring."""
    yy, xx = np.indices(plume.shape[1:], dtype=float)
    details = []
    for row in rows:
        signal = np.maximum(np.asarray(plume[row], float), 0.0)
        total = float(signal.sum())
        uu, vv, speed = _median_wind(u[row], v[row])
        dx = float(np.sum((xx - source_yx[1]) * signal) / max(total, 1e-12))
        dy = float(np.sum((yy - source_yx[0]) * signal) / max(total, 1e-12))
        cosine = (dx * uu + dy * vv) / max(np.hypot(dx, dy) * speed, 1e-12)
        operational, _ = generate_operational_template(
            signal.shape, source_yx, uu, vv, 2.0, np.ones_like(signal, bool), template_config
        )
        plume_cosine = float(np.sum(operational * signal) / max(np.linalg.norm(signal), 1e-12))
        details.append({"source_row": row, "u": uu, "v": vv, "wind_speed": speed,
                        "plume_centroid_dx_px": dx, "plume_centroid_dy_px": dy,
                        "wind_centroid_cosine": float(cosine), "template_plume_cosine": plume_cosine})
    median_alignment = float(np.median([d["wind_centroid_cosine"] for d in details]))
    return {"scope": "selected physical SMARTCARB rows; used only for coordinate audit",
            "passed": bool(median_alignment > 0), "median_wind_centroid_cosine": median_alignment,
            "rows": details}


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _prepare_h5(path: Path, n: int, shape: tuple[int, int]) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    h, w = shape; strings = h5py.string_dtype("utf-8")
    handle.create_dataset("image", (n, h, w, 3), dtype="f4", chunks=(1, h, w, 3), compression="lzf")
    handle.create_dataset("valid_mask", (n, h, w), dtype="u1", chunks=(1, h, w), compression="lzf")
    handle.create_dataset("emission", (n,), dtype="f4")
    handle.create_dataset("background_id", (n,), dtype=strings)
    handle.create_dataset("scene_id", (n,), dtype=strings)
    return handle


def _score_image(
    xco2: np.ndarray,
    valid: np.ndarray,
    u_field: np.ndarray,
    v_field: np.ndarray,
    source_yx: tuple[float, float],
    config: dict[str, Any],
    template_config: TemplateConfig,
) -> tuple[float, dict[str, Any], np.ndarray]:
    mf = config["matched_filter"]
    uu, vv, _ = _median_wind(u_field, v_field)
    residual, _ = build_residual(xco2, valid, float(mf["background_sigma_px"]))
    template, support = generate_operational_template(
        xco2.shape, source_yx, uu, vv, float(mf["grid_resolution_km"]), valid, template_config
    )
    bg_mask = make_background_reference_mask(
        xco2.shape, source_yx, uu, vv, float(mf["grid_resolution_km"]), valid,
        template_config, float(mf["plume_exclusion_sigma"]), float(mf["source_buffer_km"]),
    )
    score, quality = matched_filter_score(
        residual, template, bg_mask, str(mf["variance_estimator"]), int(mf["min_background_pixels"])
    )
    valid_values = residual[valid & np.isfinite(residual)]
    residual_std = float(np.std(valid_values)) if valid_values.size else float("nan")
    residual_mad = float(1.4826 * np.median(np.abs(valid_values - np.median(valid_values)))) if valid_values.size else float("nan")
    quality.update({"residual_std": residual_std, "residual_mad_sigma": residual_mad,
                    "template_norm": float(np.linalg.norm(template[support])),
                    "n_valid_template_pixels": int(np.count_nonzero(support & (template != 0))),
                    "background_fraction": float(np.mean(bg_mask))})
    return score, quality, residual


def build_formal_detectability_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Build realizations, freeze tau0, assign P_det, and write an HDF5/CSV pair."""
    seed = int(config["seed"]); rng = np.random.default_rng(seed)
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(config["legacy"]["background_manifest"])
    owner = validate_frozen_groups(config["split"], manifest["background_name"].astype(str))
    manifest["split_stage3"] = manifest["background_name"].map(owner)
    manifest[["background_name", "split_stage3", "category", "path"]].to_csv(output / "split_manifest.csv", index=False)

    with h5py.File(config["legacy"]["plume_dataset"], "r") as source:
        plume = np.asarray(source["plume"][:], np.float32)
        u = np.asarray(source["u"][:], np.float32)
        v = np.asarray(source["v"][:], np.float32)
        emission_ref = np.asarray(source["emiss"][:, 1], np.float32)
        point_source = _decode(source["point_source"][:])
        source_idx = np.asarray(source["idx_img"][:], int)
    selected = select_physical_rows(u, v, list(config["realizations"]["physical_wind_quantiles"]))
    template_config = TemplateConfig(**config["matched_filter"]["template"])
    source_yx = (float(config["source"]["y"]), float(config["source"]["x"]))
    feasibility = coordinate_feasibility_audit(plume, u, v, selected, source_yx, template_config)
    (output / "matched_filter_feasibility.json").write_text(json.dumps(feasibility, indent=2), encoding="utf-8")
    if not feasibility["passed"]:
        raise RuntimeError("Matched Filter feasibility failed: source/wind array convention is inconsistent")

    emissions = [float(q) for q in config["detectability"]["emission_grid_mt_yr"]]
    repetitions = int(config["realizations"]["repetitions"])
    n_total = len(manifest) * len(selected) * repetitions * (1 + len(emissions))
    dataset_path = output / "detectability_stage2.h5"
    handle = _prepare_h5(dataset_path, n_total, (32, 32))
    records: list[dict[str, Any]] = []; cursor = 0

    def write_condition(bg: pd.Series, low: np.ndarray, residual_base: np.ndarray, valid: np.ndarray,
                        wind_row: int, emission: float) -> list[int]:
        nonlocal cursor
        condition = f"{bg.background_name}|windrow={wind_row}|Q={emission:g}"
        condition_rows: list[int] = []
        ref = float(emission_ref[wind_row]); scale = 0.0 if emission == 0 else emission / ref
        condition_scores: list[float] = []
        condition_records: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            noise = spatial_block_bootstrap(residual_base, int(config["realizations"]["spatial_block_size_px"]), rng)
            xco2 = low + noise + (0.0 if emission == 0 else plume[wind_row] * scale)
            xco2 = np.where(valid, xco2, np.nan).astype(np.float32)
            score, quality, _ = _score_image(xco2, valid, u[wind_row], v[wind_row], source_yx, config, template_config)
            uu, vv, speed = _median_wind(u[wind_row], v[wind_row])
            sample_id = f"{condition}|r={repetition:03d}"
            handle["image"][cursor] = np.stack((xco2, u[wind_row], v[wind_row]), axis=-1)
            handle["valid_mask"][cursor] = valid.astype(np.uint8)
            handle["emission"][cursor] = emission
            handle["background_id"][cursor] = str(bg.background_name)
            handle["scene_id"][cursor] = sample_id
            record = {
                "row_index": cursor, "sample_id": sample_id, "condition_id": condition,
                "background_id": str(bg.background_name), "scene_id": str(bg.background_name),
                "background_type": str(bg.category), "split": owner[str(bg.background_name)],
                "emission": emission, "wind_u": uu, "wind_v": vv, "wind_speed": speed,
                "wind_direction": float(np.degrees(np.arctan2(vv, uu))), "noise_level": quality["residual_mad_sigma"],
                "valid_fraction": float(np.mean(valid)), "matched_filter_score": score,
                "background_sigma": quality["sigma_background"], "sigma_residual": quality["residual_std"],
                "residual_std": quality["residual_std"], "residual_mad_sigma": quality["residual_mad_sigma"],
                "n_background_pixels": quality["background_pixels"], "background_fraction": quality["background_fraction"],
                "quality_flag": quality["quality_flag"], "template_norm": quality["template_norm"],
                "n_valid_template_pixels": quality["n_valid_template_pixels"],
                "source_x": source_yx[1], "source_y": source_yx[0],
                "distance_source_to_edge_km": distance_source_to_edge_km(source_yx, xco2.shape, float(config["matched_filter"]["grid_resolution_km"])),
                "reference_emission": ref if emission > 0 else np.nan,
                "plume_scaling_factor": scale if emission > 0 else 0.0,
                "source_library_row": wind_row, "source_idx_img": int(source_idx[wind_row]),
                "point_source": point_source[wind_row],
                "wind_experiment_type": config["realizations"]["wind_experiment_type"],
                "repetition": repetition, "n_repetitions": repetitions,
            }
            condition_scores.append(score); condition_records.append(record); condition_rows.append(cursor); cursor += 1
        if emission > 0:
            condition_records[0]["_condition_scores"] = condition_scores
        records.extend(condition_records)
        return condition_rows

    # Generate nulls first so tau0 is frozen before positive labels are assigned.
    for bg in manifest.itertuples(index=False):
        path = Path(config["legacy"]["repo"]) / str(bg.path)
        with np.load(path) as data:
            low = np.asarray(data["background_low"], np.float32)
            residual_base = np.asarray(data["background_residual"], np.float32)
            valid = np.isfinite(np.asarray(data["xco2_grid"], float))
        for wind_row in selected:
            write_condition(bg, low, residual_base, valid, wind_row, 0.0)

    fit_parts = set(config["detectability"]["tau0_fit_partitions"])
    if "test" in fit_parts or "calibration" in fit_parts:
        raise ValueError("tau0 may use Train/Validation Class-0 only")
    fit_scores = [r["matched_filter_score"] for r in records if r["split"] in fit_parts]
    tau0 = fit_false_alarm_threshold(fit_scores, float(config["detectability"]["false_alarm_rate"]))

    for bg in manifest.itertuples(index=False):
        path = Path(config["legacy"]["repo"]) / str(bg.path)
        with np.load(path) as data:
            low = np.asarray(data["background_low"], np.float32)
            residual_base = np.asarray(data["background_residual"], np.float32)
            valid = np.isfinite(np.asarray(data["xco2_grid"], float))
        for wind_row in selected:
            for emission in emissions:
                start = len(records)
                write_condition(bg, low, residual_base, valid, wind_row, emission)
                scores = records[start]["_condition_scores"]
                p_det = detection_probability(scores, tau0)
                p_det_se = float(np.sqrt(p_det * (1.0 - p_det) / repetitions))
                for row in records[start:]:
                    row.pop("_condition_scores", None); row["p_det"] = p_det; row["p_det_se"] = p_det_se
    handle.close()
    if cursor != n_total:
        raise RuntimeError(f"Wrote {cursor} rows, expected {n_total}")

    threshold = float(config["detectability"]["quantifiable_threshold"])
    for row in records:
        row["tau0"] = tau0
        row["presence_target"] = float(row["emission"] > 0)
        if row["emission"] <= 0:
            row["p_det"] = 0.0; row["p_det_se"] = 0.0
        row["hard_class"] = hard_class(row["emission"], row["p_det"], threshold)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "detectability_manifest.csv", index=False)

    calibration = frame[(frame.emission == 0) & frame.split.isin(fit_parts)]
    heldout = frame[(frame.emission == 0) & (frame.split == "test")]
    hash_value = _config_hash(config["matched_filter"])
    tau_payload = {
        "tau0": tau0, "false_alarm_target": config["detectability"]["false_alarm_rate"],
        "fit_partitions": sorted(fit_parts), "number_of_class0_realizations": int(len(calibration)),
        "background_ids_used": sorted(calibration.background_id.unique().tolist()),
        "calibration_empirical_fpr": float(np.mean(calibration.matched_filter_score > tau0)),
        "heldout_test_empirical_fpr": float(np.mean(heldout.matched_filter_score > tau0)),
        "heldout_test_n": int(len(heldout)), "matched_filter_config_hash": hash_value,
    }
    (output / "tau0_calibration.json").write_text(json.dumps(tau_payload, indent=2), encoding="utf-8")
    sensitivity = []
    for cut in config["detectability"]["sensitivity_thresholds"]:
        classes = np.where(frame.emission <= 0, 0, np.where(frame.p_det < float(cut), 1, 2))
        counts = pd.Series(classes).value_counts()
        sensitivity.append({"threshold": cut, "class0": int(counts.get(0, 0)),
                            "class1": int(counts.get(1, 0)), "class2": int(counts.get(2, 0))})
    pd.DataFrame(sensitivity).to_csv(output / "class_balance_threshold_sensitivity.csv", index=False)
    return {"dataset_path": str(dataset_path), "manifest_path": str(output / "detectability_manifest.csv"),
            "tau": tau_payload, "feasibility": feasibility, "n_rows": len(frame),
            "selected_physical_rows": selected}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
