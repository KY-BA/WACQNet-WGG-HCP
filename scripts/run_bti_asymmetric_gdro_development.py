"""Development-only training and validation of asymmetric BTI-AGDRO.

The 25-group training pool is used for background LOBO.  Eleven previously
opened Calibration groups are explicitly downgraded to external Development
validation.  No Confirmatory Test predictions or metrics are read.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bti_asymmetric_gdro import (
    AsymmetricFit,
    asymmetric_scales,
    asymmetric_scores,
    fit_asymmetric_gdro,
    prepare_asymmetric_features,
)
from src.bti_gdro import deployable_scale, prepare_gdro_features
from src.bti_gdro_frozen_pipeline import fit_from_artifact


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(seed: int, value: str) -> int:
    token = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def verify_protocol(config_path: Path, cfg: dict) -> dict[str, str]:
    audit_path = ROOT / "outputs/bti_asymmetric_gdro_protocol/protocol_freeze_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = {
        "config_sha256": sha256(config_path),
        "development_script_sha256": sha256(Path(__file__)),
        "algorithm_module_sha256": sha256(ROOT / "src/bti_asymmetric_gdro.py"),
    }
    for key, observed in checks.items():
        if audit.get(key) != observed:
            raise RuntimeError(f"Asymmetric BTI protocol integrity failure: {key}")
    if audit.get("frozen_before_training") is not True:
        raise RuntimeError("Asymmetric BTI protocol was not frozen before training")
    if audit.get("confirmatory_test_rows_read_at_freeze") != 0:
        raise RuntimeError("Confirmatory Test leakage before asymmetric development")
    return checks


def parse_input_wind_speed(frame: pd.DataFrame) -> np.ndarray:
    result = frame.true_wind_speed.to_numpy(float).copy()
    is_speed_error = frame.experiment.astype(str).eq("wind_speed_input_error").to_numpy()
    parameters = frame.condition_id.astype(str).str.extract(
        r"\|parameter=([-+0-9.eE]+)(?:\||$)"
    )[0]
    factors = pd.to_numeric(parameters, errors="coerce").to_numpy(float)
    if np.isnan(factors[is_speed_error]).any():
        raise RuntimeError("Cannot reconstruct input wind speed")
    result[is_speed_error] *= factors[is_speed_error]
    return result


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    policy = cfg["development_data"]
    paths = {
        "original": Path(policy["training_original_13"]),
        "old_calibration": Path(policy["training_demoted_old_calibration_12"]),
        "old_manifest": Path(policy["training_old_split_manifest"]),
        "external": Path(policy["external_development_demoted_v2_calibration_11"]),
        "v2_manifest": Path(policy["v2_split_manifest"]),
    }
    expected = {
        "original": policy["training_original_13_sha256"],
        "old_calibration": policy["training_demoted_old_calibration_12_sha256"],
        "old_manifest": policy["training_old_split_manifest_sha256"],
        "external": policy["external_development_demoted_v2_calibration_11_sha256"],
        "v2_manifest": policy["v2_split_manifest_sha256"],
    }
    observed = {key: sha256(path) for key, path in paths.items()}
    if observed != expected:
        mismatch = [key for key in observed if observed[key] != expected[key]]
        raise RuntimeError(f"Development source integrity failure: {mismatch}")

    original = pd.read_csv(paths["original"])
    old_calibration = pd.read_csv(paths["old_calibration"])
    old_manifest = pd.read_csv(paths["old_manifest"])
    external = pd.read_csv(paths["external"])
    v2_manifest = pd.read_csv(paths["v2_manifest"])

    old_cal_manifest = old_manifest.loc[old_manifest.role.eq("bti_calibration")].copy()
    old_test_ids = set(old_manifest.loc[old_manifest.role.eq("bti_confirmatory_test"), "background_id"].astype(str))
    if set(old_calibration.background_id.astype(str)) != set(old_cal_manifest.background_id.astype(str)):
        raise RuntimeError("Old demoted Calibration does not match its frozen role")
    old_metadata = old_cal_manifest.set_index("background_id")
    old_calibration = old_calibration.copy()
    old_calibration["background_sigma"] = old_calibration.background_id.map(old_metadata.background_sigma)
    old_calibration["input_wind_speed"] = parse_input_wind_speed(old_calibration)

    v2_cal_manifest = v2_manifest.loc[v2_manifest.role.eq("new_calibration")].copy()
    v2_test_ids = set(v2_manifest.loc[v2_manifest.role.eq("new_confirmatory_test"), "background_id"].astype(str))
    if set(external.background_id.astype(str)) != set(v2_cal_manifest.background_id.astype(str)):
        raise RuntimeError("External Development does not match downgraded v2 Calibration")
    if not external.role.astype(str).eq("new_calibration").all():
        raise RuntimeError("Non-Calibration role in external Development source")

    original = original.copy()
    old_calibration = old_calibration.copy()
    external = external.copy()
    original["development_scope"] = "original_development_13"
    old_calibration["development_scope"] = "demoted_old_calibration_12"
    external["development_scope"] = "demoted_v2_calibration_11"
    common = [
        "sample_id", "condition_id", "background_id", "background_type",
        "development_scope", "true_emission", "mu_fixed", "background_sigma",
        "valid_fraction", "retained_valid_fraction", "input_wind_speed",
    ]
    training = pd.concat([original[common], old_calibration[common]], ignore_index=True)
    external = external[common].copy()
    training = prepare_asymmetric_features(training)
    external = prepare_asymmetric_features(external)

    if training.background_id.nunique() != int(policy["expected_training_groups"]):
        raise RuntimeError("Unexpected training-group count")
    if external.background_id.nunique() != int(policy["expected_external_development_groups"]):
        raise RuntimeError("Unexpected external Development group count")
    training_ids = set(training.background_id.astype(str))
    external_ids = set(external.background_id.astype(str))
    forbidden = old_test_ids | v2_test_ids
    if training_ids & external_ids:
        raise RuntimeError("Training/external Development overlap")
    if (training_ids | external_ids) & forbidden:
        raise RuntimeError("Confirmatory Test leakage into asymmetric development")

    audit = {
        "training_groups": int(training.background_id.nunique()),
        "training_rows": int(len(training)),
        "external_development_groups": int(external.background_id.nunique()),
        "external_development_rows": int(len(external)),
        "development_groups_total": int(pd.concat([training, external]).background_id.nunique()),
        "v2_confirmatory_test_groups_read": 0,
        "v2_confirmatory_test_rows_read": 0,
        "historical_confirmatory_test_groups_read": 0,
        "historical_confirmatory_test_rows_read": 0,
        "source_hashes": observed,
    }
    return training, external, audit


def equal_group_sample(frame: pd.DataFrame, rows_per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    for background, part in frame.groupby("background_id", sort=True):
        n = min(rows_per_group, len(part))
        rng = np.random.default_rng(stable_seed(seed, str(background)))
        positions = np.sort(rng.choice(len(part), size=n, replace=False))
        parts.append(part.iloc[positions])
    return pd.concat(parts, ignore_index=True)


def hcp_quantile(groups: list[np.ndarray], coverage: float) -> float:
    k_groups = len(groups)
    finite_mass = k_groups / (k_groups + 1.0)
    if coverage > finite_mass + 1e-15:
        return float("inf")
    values = np.concatenate(groups)
    weights = np.concatenate([
        np.full(len(group), 1.0 / ((k_groups + 1.0) * len(group)))
        for group in groups
    ])
    order = np.argsort(values, kind="stable")
    index = int(np.searchsorted(np.cumsum(weights[order]), coverage, side="left"))
    return float(values[order[min(index, len(order) - 1)]])


def grouped_qhat(frame: pd.DataFrame, scores: np.ndarray, coverage: float) -> float:
    values = pd.Series(np.asarray(scores, float), index=frame.index)
    groups = [
        values.loc[part.index].to_numpy(float)
        for _, part in frame.groupby("background_id", sort=True)
    ]
    return hcp_quantile(groups, coverage)


def evaluate_asymmetric(
    frame: pd.DataFrame,
    scale_minus: np.ndarray,
    scale_plus: np.ndarray,
    qhat: float,
    nominal: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    truth, mu = frame.true_emission.to_numpy(float), frame.mu_fixed.to_numpy(float)
    lower = np.maximum(0.0, mu - qhat * scale_minus)
    upper = mu + qhat * scale_plus
    width = upper - lower
    covered = (truth >= lower) & (truth <= upper)
    score = asymmetric_scores(frame, scale_minus, scale_plus)
    picp = float(np.mean(covered))
    metrics = {
        "n": int(len(frame)),
        "picp": picp,
        "coverage_error": abs(picp - nominal),
        "undercoverage": max(0.0, nominal - picp),
        "mpiw": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "median_normalized_width": float(np.median(width / (np.maximum(mu, 0.0) + 1.0))),
        "p90_width": float(np.quantile(width, 0.90)),
        "p95_width": float(np.quantile(width, 0.95)),
        "lower_zero_fraction": float(np.mean(lower == 0.0)),
    }
    predictions = pd.DataFrame({
        "sample_id": frame.sample_id.to_numpy(),
        "background_id": frame.background_id.to_numpy(),
        "true_emission": truth,
        "mu_fixed": mu,
        "scale_minus": scale_minus,
        "scale_plus": scale_plus,
        "score": score,
        "lower90": lower,
        "upper90": upper,
        "width90": width,
        "covered90": covered,
    })
    return metrics, predictions


def evaluate_symmetric(
    frame: pd.DataFrame, scale: np.ndarray, qhat: float, nominal: float
) -> tuple[dict[str, float | int], pd.DataFrame]:
    return evaluate_asymmetric(frame, scale, scale, qhat, nominal)


def fit_candidate(frame: pd.DataFrame, candidate: dict, cfg: dict) -> AsymmetricFit:
    opt = cfg["optimization"]
    features = cfg["features"]["full"] if candidate["features"] == "full" else candidate["features"]
    return fit_asymmetric_gdro(
        frame,
        method=candidate["id"],
        feature_names=features,
        tau=float(opt["pinball_tau"]),
        dro_temperature=float(opt["dro_temperature"]),
        tail_quantile=float(opt["tail_quantile"]),
        tail_variance_weight=float(candidate["tail_variance_weight"]),
        tail_cvar_weight=float(candidate["tail_cvar_weight"]),
        tail_cvar_fraction=float(opt["tail_cvar_fraction"]),
        width_weight=float(candidate["width_weight"]),
        l2_penalty=float(opt["l2_penalty"]),
        log_mu_bounds=tuple(opt["log_mu_raw_coefficient_bounds"]),
        other_bounds=tuple(opt["other_raw_coefficient_bounds"]),
        correction_clip=float(opt["correction_clip"]),
        residual_epsilon=float(opt["residual_epsilon"]),
        maxiter=int(opt["maxiter"]),
    )


def summarize_methods(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, part in metrics.groupby("method", sort=False):
        pred = predictions.loc[predictions.method.eq(method)]
        tails = pred.groupby("background_id").score.agg(
            p90=lambda x: np.quantile(x, 0.90),
            p95=lambda x: np.quantile(x, 0.95),
        )
        rows.append({
            "method": method,
            "n_backgrounds": int(len(part)),
            "n_picp_lt_0_85": int((part.picp < 0.85).sum()),
            "n_picp_lt_0_80": int((part.picp < 0.80).sum()),
            "background_median_picp": float(part.picp.median()),
            "background_median_coverage_error": float(part.coverage_error.median()),
            "background_median_mpiw": float(part.mpiw.median()),
            "background_median_normalized_width": float(part.median_normalized_width.median()),
            "background_median_lower_zero_fraction": float(part.lower_zero_fraction.median()),
            "background_p90_score_cv": float(tails.p90.std(ddof=0) / tails.p90.mean()),
            "background_p95_score_cv": float(tails.p95.std(ddof=0) / tails.p95.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/bti_asymmetric_gdro_development.json"))
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_hashes = verify_protocol(config_path, cfg)
    training, external, access = load_data(cfg)
    access.update(protocol_hashes)
    access["run_utc"] = datetime.now(timezone.utc).isoformat()
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "access_audit.json").write_text(json.dumps(access, indent=2), encoding="utf-8")

    opt, validation = cfg["optimization"], cfg["validation"]
    candidates = list(opt["candidates"])
    nominal = float(validation["nominal_coverage"])
    clip = float(opt["correction_clip"])
    rows_per_group = int(opt["fit_rows_per_group"])
    seed = int(opt["fit_subsample_seed"])
    inner_metrics, inner_predictions, fold_parameters = [], [], []

    backgrounds = sorted(training.background_id.astype(str).unique())
    for fold, heldout in enumerate(backgrounds):
        train = training.loc[~training.background_id.astype(str).eq(heldout)].reset_index(drop=True)
        held = training.loc[training.background_id.astype(str).eq(heldout)].reset_index(drop=True)
        fit_frame = equal_group_sample(train, rows_per_group, seed)
        for candidate in candidates:
            fit = fit_candidate(fit_frame, candidate, cfg)
            train_minus, train_plus = asymmetric_scales(train, fit, clip)
            held_minus, held_plus = asymmetric_scales(held, fit, clip)
            qhat = grouped_qhat(train, asymmetric_scores(train, train_minus, train_plus), nominal)
            metrics, predictions = evaluate_asymmetric(held, held_minus, held_plus, qhat, nominal)
            method = candidate["id"]
            inner_metrics.append({
                "outer_fold": fold,
                "heldout_background": heldout,
                "development_scope": held.development_scope.iloc[0],
                "method": method,
                "qhat90": qhat,
                **metrics,
            })
            predictions["method"] = method
            predictions["outer_fold"] = fold
            inner_predictions.append(predictions)
            fold_parameters.append({
                "outer_fold": fold,
                "heldout_background": heldout,
                "method": method,
                "fit_json": json.dumps(fit.to_dict(), ensure_ascii=False, sort_keys=True),
            })
        print(f"Asymmetric BTI LOBO {fold + 1}/{len(backgrounds)}: {heldout}", flush=True)

    inner_metric_frame = pd.DataFrame(inner_metrics)
    inner_prediction_frame = pd.concat(inner_predictions, ignore_index=True)
    inner_summary = summarize_methods(inner_metric_frame, inner_prediction_frame)

    fit_frame = equal_group_sample(training, rows_per_group, seed)
    external_metrics, external_predictions, final_fits = [], [], {}
    for candidate in candidates:
        fit = fit_candidate(fit_frame, candidate, cfg)
        final_fits[candidate["id"]] = fit
        train_minus, train_plus = asymmetric_scales(training, fit, clip)
        ext_minus, ext_plus = asymmetric_scales(external, fit, clip)
        qhat = grouped_qhat(training, asymmetric_scores(training, train_minus, train_plus), nominal)
        method = candidate["id"]
        for background, part in external.groupby("background_id", sort=True):
            positions = part.index.to_numpy()
            metrics, predictions = evaluate_asymmetric(
                part,
                ext_minus[positions - external.index.min()],
                ext_plus[positions - external.index.min()],
                qhat,
                nominal,
            )
            external_metrics.append({
                "background_id": background,
                "background_type": part.background_type.iloc[0],
                "method": method,
                "qhat90": qhat,
                **metrics,
            })
            predictions["method"] = method
            external_predictions.append(predictions)

    a4_artifact = json.loads(Path(cfg["frozen_a4_reference"]["artifact"]).read_text(encoding="utf-8"))
    a4_fit = fit_from_artifact(a4_artifact)
    training_a4 = prepare_gdro_features(training, 0.89187068)
    external_a4 = prepare_gdro_features(external, 0.89187068)
    training_scale = deployable_scale(training_a4, a4_fit, 3.0)
    external_scale = deployable_scale(external_a4, a4_fit, 3.0)
    a4_qhat = grouped_qhat(training_a4, training_a4.absolute_error.to_numpy(float) / training_scale, nominal)
    for background, part in external_a4.groupby("background_id", sort=True):
        positions = part.index.to_numpy()
        metrics, predictions = evaluate_symmetric(
            part, external_scale[positions - external_a4.index.min()], a4_qhat, nominal
        )
        external_metrics.append({
            "background_id": background,
            "background_type": part.background_type.iloc[0],
            "method": "A0_FROZEN_A4_REFERENCE",
            "qhat90": a4_qhat,
            **metrics,
        })
        predictions["method"] = "A0_FROZEN_A4_REFERENCE"
        external_predictions.append(predictions)

    external_metric_frame = pd.DataFrame(external_metrics)
    external_prediction_frame = pd.concat(external_predictions, ignore_index=True)
    external_summary = summarize_methods(external_metric_frame, external_prediction_frame)

    primary_id = opt["preregistered_primary_candidate"]
    primary = external_summary.loc[external_summary.method.eq(primary_id)].iloc[0]
    reference = external_summary.loc[external_summary.method.eq("A0_FROZEN_A4_REFERENCE")].iloc[0]
    a2 = external_summary.loc[external_summary.method.eq("A2_ASYMMETRIC_GROUP_DRO")].iloc[0]
    width_reduction = 1.0 - float(primary.background_median_normalized_width) / float(reference.background_median_normalized_width)
    zero_reduction = float(reference.background_median_lower_zero_fraction) - float(primary.background_median_lower_zero_fraction)
    tail_reduction = 1.0 - float(primary.background_p95_score_cv) / float(a2.background_p95_score_cv)
    gate = validation["primary_adoption_gate"]
    gates = {
        "n_picp_lt_0_85_not_worse_than_frozen_a4": bool(primary.n_picp_lt_0_85 <= reference.n_picp_lt_0_85),
        "n_picp_lt_0_80_not_worse_than_frozen_a4": bool(primary.n_picp_lt_0_80 <= reference.n_picp_lt_0_80),
        "median_coverage_error_within_tolerance": bool(primary.background_median_coverage_error <= reference.background_median_coverage_error + float(gate["median_coverage_error_tolerance_vs_a4"])),
        "median_normalized_width_reduction_pass": bool(width_reduction >= float(gate["minimum_median_normalized_width_reduction_vs_a4"])),
        "lower_zero_fraction_reduction_pass": bool(zero_reduction >= float(gate["minimum_lower_zero_fraction_reduction_vs_a4"])),
        "p95_tail_cv_reduction_vs_a2_pass": bool(tail_reduction >= float(gate["minimum_p95_tail_cv_reduction_vs_a2"])),
    }
    established = bool(all(gates.values()))
    verdict = "BTI_AGDRO_DEVELOPMENT_ESTABLISHED" if established else "BTI_AGDRO_DEVELOPMENT_NOT_ESTABLISHED"
    selection = {
        "stage": cfg["stage"],
        "preregistered_primary_candidate": primary_id,
        "verdict": verdict,
        "primary_candidate_adopted": established,
        "external_development_width_reduction_vs_a4": width_reduction,
        "external_development_lower_zero_reduction_vs_a4": zero_reduction,
        "external_development_p95_tail_cv_reduction_vs_a2": tail_reduction,
        "gates": gates,
        "v2_confirmatory_test_used": False,
        "historical_confirmatory_test_used": False,
        "post_confirmatory_tuning_performed": False,
    }

    inner_metric_frame.to_csv(out / "inner_lobo_background_metrics.csv", index=False)
    inner_prediction_frame.to_csv(out / "inner_lobo_predictions.csv", index=False)
    pd.DataFrame(fold_parameters).to_csv(out / "inner_lobo_parameters.csv", index=False)
    inner_summary.to_csv(out / "inner_lobo_method_summary.csv", index=False)
    external_metric_frame.to_csv(out / "external_development_background_metrics.csv", index=False)
    external_prediction_frame.to_csv(out / "external_development_predictions.csv", index=False)
    external_summary.to_csv(out / "external_development_method_summary.csv", index=False)
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    if established:
        combined = pd.concat([training, external], ignore_index=True)
        final_fit = fit_candidate(equal_group_sample(combined, rows_per_group, seed), next(c for c in candidates if c["id"] == primary_id), cfg)
        artifact = {
            **selection,
            "algorithm": "Background-Tail-Invariant Asymmetric Group-DRO Hierarchical Conformal Prediction",
            "score_formula": "max((Q_hat-Q_true)_+/s_minus(x),(Q_true-Q_hat)_+/s_plus(x))",
            "interval_formula": "[max(0,Q_hat-q*s_minus(x)), Q_hat+q*s_plus(x)]",
            "deployment_features": cfg["features"]["full"],
            "forbidden_deployment_features": cfg["features"]["forbidden"],
            "fit": final_fit.to_dict(),
            "development_background_count": int(combined.background_id.nunique()),
            "formal_qhat_fitted": False,
            "new_calibration_required": True,
            "new_confirmatory_test_required": True,
            "config_sha256": sha256(config_path),
        }
        artifact_path = out / "bti_agdro_frozen.json"
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        (out / "bti_agdro_frozen.sha256").write_text(sha256(artifact_path) + "\n", encoding="ascii")

    report = [
        "# BTI-AGDRO Development结果",
        "",
        f"- 判定：**{verdict}**",
        "- 训练/内部LOBO：25个Development背景；外部Development验证：11个明确降级Calibration背景。",
        "- 所有Confirmatory Test：未读取、未用于选择。",
        "- A3是预注册主候选；A1/A2是机制消融；A0是冻结A4参考。",
        "",
        "## 25-background LOBO",
        "",
        inner_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 11-background外部Development验证",
        "",
        external_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## A3预注册Gate",
        "",
        f"- normalized width reduction vs A4: {width_reduction:.6f}",
        f"- lower-zero reduction vs A4: {zero_reduction:.6f}",
        f"- P95 tail-CV reduction vs A2: {tail_reduction:.6f}",
        f"- gates: `{json.dumps(gates, ensure_ascii=False)}`",
        "",
        "该结果仅为Development级算法选择，不是新的Confirmatory结论。",
    ]
    (out / "report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "selection": selection,
        "inner_summary": inner_summary.to_dict("records"),
        "external_summary": external_summary.to_dict("records"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
