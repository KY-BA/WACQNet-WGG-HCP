"""Build the immutable 36-background WACQNet Level-B tensor cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bti_v2_levelb import _legacy_symbols, _median_wind, stable_seed
from src.wacqnet import build_spatial_features


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(cfg: dict) -> pd.DataFrame:
    frames = []
    for key in ["original_development_13", "demoted_historical_calibration_12", "demoted_v2_calibration_11"]:
        frame = pd.read_csv(cfg["data"][key])
        required = ["sample_id", "condition_id", "background_id", "true_emission", "mu_fixed"]
        missing = [col for col in required if col not in frame]
        if missing:
            raise RuntimeError(f"{key} missing columns {missing}")
        frames.append(frame[required + [col for col in ["background_type"] if col in frame]].copy())
    result = pd.concat(frames, ignore_index=True)
    if result.sample_id.duplicated().any():
        raise RuntimeError("Duplicate sample IDs in Development inputs")
    if result.background_id.nunique() != int(cfg["data"]["expected_backgrounds"]):
        raise RuntimeError("Development group count drift")
    counts = result.groupby("background_id").size()
    if not (counts == int(cfg["data"]["expected_rows_per_background"])).all():
        raise RuntimeError(f"Per-background count drift: {counts.to_dict()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="D:/CO2/_third/configs/wacqnet_development_protocol_v1.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    protocol_audit = json.loads((Path(cfg["outputs"]["protocol_dir"]) / "protocol_freeze_audit.json").read_text(encoding="utf-8"))
    if protocol_audit["status"] != "PASS" or protocol_audit["protocol_version"] != cfg["protocol_version"]:
        raise RuntimeError("Protocol freeze audit missing or stale")

    out = Path(cfg["outputs"]["cache_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "wacqnet_levelb_cache.h5"
    metadata_path = out / "wacqnet_levelb_metadata.csv"
    audit_path = out / "wacqnet_levelb_cache_audit.json"
    if cache_path.exists() and not args.overwrite:
        print(f"Cache already exists: {cache_path}")
        return

    rows = load_rows(cfg)
    lookup = rows.set_index("sample_id")
    scenes = pd.read_csv(cfg["data"]["scene_table"]).sort_values("background_id").reset_index(drop=True)
    type_map = rows.groupby("background_id").background_type.first().to_dict()
    symbols = _legacy_symbols(cfg["legacy_project_read_only"])
    _, select_physical_rows, spatial_block_bootstrap, TemplateConfig, generate_template, block_mask, downwind_mask, rotate_wind = symbols
    level = cfg["level_b"]
    with h5py.File(level["plume_dataset"], "r") as source:
        plume = np.asarray(source["plume"][:], np.float32)
        u_all = np.asarray(source["u"][:], np.float32)
        v_all = np.asarray(source["v"][:], np.float32)
        emission_ref = np.asarray(source["emiss"][:, 1], np.float32)
    physical_rows = select_physical_rows(u_all, v_all, list(level["physical_wind_quantiles"]))
    if physical_rows != list(level["expected_physical_rows"]):
        raise RuntimeError(f"Physical rows drift: {physical_rows}")
    detector = yaml.safe_load(Path(level["detector_config"]).read_text(encoding="utf-8"))["detector"]
    template_cfg = TemplateConfig(**detector["template"])
    source_yx = (float(detector["source_y"]), float(detector["source_x"]))
    emissions = [float(v) for v in level["emission_grid_mt_yr"]]
    core = {float(v) for v in level["core_emissions_mt_yr"]}
    repetitions = int(level["repetitions"])
    total = len(rows)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    metadata: list[dict] = []

    with h5py.File(cache_path, "w") as cache:
        ds_spatial = cache.create_dataset("spatial", (total, 16, 16, 6), dtype="float16", chunks=(64, 16, 16, 6), compression="lzf")
        ds_scalar = cache.create_dataset("scalar", (total, 3), dtype="float32", chunks=(1024, 3), compression="lzf")
        ds_y = cache.create_dataset("true_emission", (total,), dtype="float32", chunks=(4096,), compression="lzf")
        ds_group = cache.create_dataset("group_index", (total,), dtype="int16", chunks=(4096,), compression="lzf")
        ds_rep = cache.create_dataset("repetition", (total,), dtype="int16", chunks=(4096,), compression="lzf")
        ds_sid = cache.create_dataset("sample_id", (total,), dtype=string_dtype)
        write_index = 0
        seen: set[str] = set()

        def emit(bg_id: str, group_index: int, image: np.ndarray, valid: np.ndarray, sample_id: str, condition_id: str, experiment: str, missingness: str, repetition: int) -> None:
            nonlocal write_index
            if sample_id in seen:
                raise RuntimeError(f"Duplicate generated sample {sample_id}")
            if sample_id not in lookup.index:
                raise RuntimeError(f"Generated sample absent from frozen predictions: {sample_id}")
            row = lookup.loc[sample_id]
            uu, vv, speed, _ = _median_wind(image[..., 1], image[..., 2])
            template, _ = generate_template(valid.shape, source_yx, uu, vv, float(detector["grid_resolution_km"]), valid, template_cfg)
            spatial, observed = build_spatial_features(image, valid, template, source_yx)
            anchor = max(float(row.mu_fixed), 0.0)
            ds_spatial[write_index] = spatial.astype(np.float16)
            ds_scalar[write_index] = np.asarray([np.log1p(anchor), observed[0], observed[1]], np.float32)
            ds_y[write_index] = float(row.true_emission)
            ds_group[write_index] = group_index
            ds_rep[write_index] = repetition
            ds_sid[write_index] = sample_id
            metadata.append({
                "cache_index": write_index, "sample_id": sample_id, "condition_id": condition_id,
                "background_id": bg_id, "background_type": type_map.get(bg_id, "unknown"),
                "experiment": experiment, "missingness_type": missingness, "repetition": repetition,
                "true_emission": float(row.true_emission), "mu_fixed": float(row.mu_fixed),
                "input_wind_speed": speed, "valid_fraction": float(valid.mean()),
            })
            seen.add(sample_id)
            write_index += 1

        for group_index, bg in enumerate(scenes.itertuples(index=False)):
            bg_id = str(bg.background_id)
            with np.load(bg.scene_file) as scene:
                low = np.asarray(scene["background_low"], np.float32)
                residual_base = np.asarray(scene["background_residual"], np.float32)
                if "valid_mask" in scene.files:
                    natural_valid = np.asarray(scene["valid_mask"], bool)
                else:
                    neighbor = np.asarray(scene["neighbor_count"], np.float32)
                    natural_valid = (
                        np.isfinite(residual_base)
                        & np.isfinite(low)
                        & np.isfinite(np.asarray(scene["xco2_grid"], np.float32))
                        & (neighbor > 0)
                    )
            base_images: dict[tuple[float, int], np.ndarray] = {}
            for wind_row in physical_rows:
                true_u, true_v = u_all[wind_row], v_all[wind_row]
                for emission in emissions:
                    condition = f"{bg_id}|windrow={wind_row}|Q={emission:g}"
                    scaling = emission / float(emission_ref[wind_row])
                    for repetition in range(repetitions):
                        rng = np.random.default_rng(stable_seed(int(level["generation_seed"]), bg_id, wind_row, emission, repetition))
                        noise = spatial_block_bootstrap(residual_base, int(level["spatial_block_size_px"]), rng)
                        xco2 = np.where(natural_valid, low + noise + plume[wind_row] * scaling, np.nan).astype(np.float32)
                        image = np.stack((xco2, true_u, true_v), axis=-1).astype(np.float32)
                        if wind_row == int(level["base_physical_wind_row"]) and emission in core:
                            base_images[(emission, repetition)] = image.copy()
                        sid = f"{condition}|r={repetition:03d}"
                        emit(bg_id, group_index, image, natural_valid, sid, condition, "physical_wind", "natural_oco3_mask", repetition)
            base_row = int(level["base_physical_wind_row"])
            for emission in sorted(core):
                true_u, true_v = u_all[base_row], v_all[base_row]
                for repetition in range(repetitions):
                    base_image = base_images[(emission, repetition)]
                    interventions = [("wind_direction_error", float(v), "none") for v in level["wind_direction_errors_deg"] if float(v) != 0.0]
                    interventions += [("wind_speed_input_error", float(v), "none") for v in level["wind_speed_multipliers"] if float(v) != 1.0]
                    interventions += [("missingness", float(retained), str(kind)) for kind in level["missingness_types"] for retained in level["retained_valid_fractions"] if float(retained) != 1.0]
                    for experiment, parameter, missingness in interventions:
                        image = base_image.copy()
                        valid = natural_valid.copy()
                        if experiment == "wind_direction_error":
                            image[..., 1], image[..., 2] = rotate_wind(true_u, true_v, parameter, 1.0)
                        elif experiment == "wind_speed_input_error":
                            image[..., 1], image[..., 2] = rotate_wind(true_u, true_v, 0.0, parameter)
                        else:
                            key = f"{bg_id}|Q={emission:g}|r={repetition}|{missingness}|{parameter:g}"
                            if missingness == "block_correlated":
                                valid = block_mask(natural_valid, parameter, int(level["missingness_block_size_px"]), key, int(level["generation_seed"]))
                            else:
                                base_template, _ = generate_template(natural_valid.shape, source_yx, float(np.nanmedian(true_u)), float(np.nanmedian(true_v)), float(detector["grid_resolution_km"]), natural_valid, template_cfg)
                                valid = downwind_mask(natural_valid, base_template, parameter)
                            image[..., 0] = np.where(valid, image[..., 0], np.nan)
                        condition = f"{bg_id}|Q={emission:g}|windrow={base_row}|{experiment}|parameter={parameter:g}|missing={missingness}"
                        sid = f"{condition}|r={repetition:03d}"
                        emit(bg_id, group_index, image, valid, sid, condition, experiment, missingness, repetition)
            print(f"cached {group_index + 1}/{len(scenes)} {bg_id}: {write_index}/{total}", flush=True)

        if write_index != total or len(seen) != total:
            raise RuntimeError(f"Cache row mismatch: wrote {write_index}, unique {len(seen)}, expected {total}")
        missing_ids = set(lookup.index) - seen
        if missing_ids:
            raise RuntimeError(f"Frozen predictions not regenerated: {list(sorted(missing_ids))[:3]}")
        cache.attrs["protocol_sha256"] = protocol_audit["protocol_sha256"]
        cache.attrs["protocol_version"] = cfg["protocol_version"]
        cache.attrs["backgrounds"] = 36
        cache.attrs["rows"] = total

    meta = pd.DataFrame(metadata)
    meta.to_csv(metadata_path, index=False)
    audit = {
        "status": "PASS", "protocol_sha256": protocol_audit["protocol_sha256"],
        "backgrounds": int(meta.background_id.nunique()), "rows": int(len(meta)),
        "conditions_per_background": {str(k): int(v) for k, v in meta.groupby("background_id").condition_id.nunique().items()},
        "rows_per_background": {str(k): int(v) for k, v in meta.groupby("background_id").size().items()},
        "cache_path": str(cache_path), "cache_sha256": sha256(cache_path),
        "metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path),
        "current_v3_calibration_rows_read": 0, "current_v3_confirmatory_test_rows_read": 0,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
