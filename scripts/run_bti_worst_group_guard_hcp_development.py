"""Preregistered Development-only LOBO for BTI-WorstGroupGuard-HCP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_bti_asymmetric_gdro_development import (
    evaluate_symmetric,
    grouped_qhat,
    load_data,
    summarize_methods,
)
from src.bti_gdro import prepare_gdro_features
from src.bti_gdro_frozen_pipeline import fit_from_artifact
from src.bti_group_tolerance_hcp import group_tolerance_qhat, guarantee_scope
from src.bti_settail_hcp import deployable_scale_elementwise


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_protocol(config_path: Path) -> dict[str, str]:
    audit_path = ROOT / "outputs/bti_worst_group_guard_hcp_protocol/protocol_freeze_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = {
        "config_sha256": sha256(config_path),
        "algorithm_module_sha256": sha256(ROOT / "src/bti_group_tolerance_hcp.py"),
        "development_script_sha256": sha256(Path(__file__)),
        "mathematical_protocol_sha256": sha256(
            ROOT / "outputs/bti_worst_group_guard_hcp_protocol/mathematical_protocol.md"
        ),
    }
    for key, value in checks.items():
        if audit.get(key) != value:
            raise RuntimeError(f"WorstGroupGuard protocol integrity failure: {key}")
    if audit.get("frozen_before_development_evaluation") is not True:
        raise RuntimeError("Protocol was not frozen before evaluation")
    if audit.get("confirmatory_test_rows_read_at_freeze") != 0:
        raise RuntimeError("Confirmatory Test leakage at freeze")
    return checks


def load_development(cfg: dict):
    parent = json.loads(Path(cfg["base_protocol"]).read_text(encoding="utf-8"))
    base_cfg = json.loads(
        Path(parent["development_data"]["base_asymmetric_protocol"]).read_text(encoding="utf-8")
    )
    original, demoted, access = load_data(base_cfg)
    development = pd.concat([original, demoted], ignore_index=True)
    if development.background_id.nunique() != 36:
        raise RuntimeError("Expected exactly 36 Development groups")
    base = parent["frozen_base_scale"]
    artifact = Path(base["artifact"])
    if sha256(artifact) != base["artifact_sha256"]:
        raise RuntimeError("Frozen A4 artifact drift")
    a4_fit = fit_from_artifact(json.loads(artifact.read_text(encoding="utf-8")))
    development = prepare_gdro_features(development, float(base["mean_power_b"]))
    scale = deployable_scale_elementwise(development, a4_fit, float(base["scale_clip"]))
    return development, scale, access, base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs/bti_worst_group_guard_hcp_development.json")
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = verify_protocol(config_path)
    development, a4_scale, access, base = load_development(cfg)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    access.update(hashes)
    access.update({
        "development_groups_read": 36,
        "development_rows_read": int(len(development)),
        "confirmatory_test_groups_read": 0,
        "confirmatory_test_rows_read": 0,
        "hpr_training_calls": 0,
        "hpr_inference_calls": 0,
        "downloads": 0,
        "run_utc": datetime.now(timezone.utc).isoformat(),
    })
    (out / "access_audit.json").write_text(json.dumps(access, indent=2), encoding="utf-8")

    alg = cfg["algorithm"]
    nominal = float(cfg["validation"]["nominal_coverage"])
    metric_rows = []
    prediction_parts = []
    tail_parts = []
    qhat_rows = []
    ids = sorted(set(development.background_id.astype(str)))
    for fold, heldout in enumerate(ids):
        train_mask = ~development.background_id.astype(str).eq(heldout)
        held_mask = ~train_mask
        train = development.loc[train_mask].reset_index(drop=True)
        held = development.loc[held_mask].reset_index(drop=True)
        train_scale = a4_scale[train_mask.to_numpy()]
        held_scale = a4_scale[held_mask.to_numpy()]
        scores = train.absolute_error.to_numpy(float) / train_scale

        q0 = grouped_qhat(train, scores, nominal)
        metrics, predictions = evaluate_symmetric(held, held_scale, q0, nominal)
        metric_rows.append({
            "outer_fold": fold, "heldout_background": heldout,
            "method": "A4_FROZEN_REFERENCE", "qhat90": q0, **metrics,
        })
        predictions["method"] = "A4_FROZEN_REFERENCE"
        predictions["outer_fold"] = fold
        prediction_parts.append(predictions)

        result, tails = group_tolerance_qhat(
            train,
            scores,
            within_group_coverage=float(alg["within_group_coverage"]),
            target_group_success_probability=float(alg["target_group_success_probability"]),
        )
        if result.group_rank != 35 or not np.isclose(result.qhat, tails.tail_threshold.max()):
            raise RuntimeError("Worst-group guard did not select the maximum training group tail")
        metrics, predictions = evaluate_symmetric(held, held_scale, result.qhat, nominal)
        metric_rows.append({
            "outer_fold": fold, "heldout_background": heldout,
            "method": "BTI_WORST_GROUP_GUARD_HCP", "qhat90": result.qhat,
            "between_group_rank": result.group_rank,
            "guaranteed_group_success_probability": result.guaranteed_group_success_probability,
            **metrics,
        })
        predictions["method"] = "BTI_WORST_GROUP_GUARD_HCP"
        predictions["outer_fold"] = fold
        prediction_parts.append(predictions)
        tails["outer_fold"] = fold
        tails["outer_heldout_background"] = heldout
        tail_parts.append(tails)
        qhat_rows.append({"outer_fold": fold, "heldout_background": heldout, **result.to_dict()})
        print(f"BTI-WorstGroupGuard-HCP LOBO {fold + 1}/36: {heldout}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    tails = pd.concat(tail_parts, ignore_index=True)
    qhats = pd.DataFrame(qhat_rows)
    summary = summarize_methods(metrics, predictions)
    metrics.to_csv(out / "lobo_background_metrics.csv", index=False)
    predictions.to_csv(out / "lobo_predictions.csv", index=False)
    tails.to_csv(out / "outer_training_group_tails.csv", index=False)
    qhats.to_csv(out / "lobo_worst_group_guard_qhats.csv", index=False)
    summary.to_csv(out / "lobo_method_summary.csv", index=False)

    primary = summary.loc[summary.method.eq("BTI_WORST_GROUP_GUARD_HCP")].iloc[0]
    reference = summary.loc[summary.method.eq("A4_FROZEN_REFERENCE")].iloc[0]
    pmetrics = metrics.loc[metrics.method.eq("BTI_WORST_GROUP_GUARD_HCP")]
    groups_ge90 = int((pmetrics.picp >= .90).sum())
    width_ratio = float(
        primary.background_median_normalized_width /
        reference.background_median_normalized_width
    )
    val = cfg["validation"]
    gates = {
        "heldout_group_target_pass": groups_ge90 >= int(
            val["minimum_heldout_groups_picp_at_least_0_90"]
        ),
        "n_picp_lt_0_85_pass": int(primary.n_picp_lt_0_85) <= int(
            val["maximum_n_picp_below_0_85"]
        ),
        "n_picp_lt_0_80_pass": int(primary.n_picp_lt_0_80) <= int(
            val["maximum_n_picp_below_0_80"]
        ),
        "worst_background_picp_pass": float(pmetrics.picp.min()) >= float(
            val["minimum_worst_background_picp"]
        ),
        "width_cost_pass": width_ratio <= float(
            val["maximum_median_normalized_width_ratio_vs_a4"]
        ),
    }
    established = all(gates.values())
    verdict = (
        "BTI_WORST_GROUP_GUARD_HCP_ESTABLISHED_ON_DEVELOPMENT"
        if established else "BTI_WORST_GROUP_GUARD_HCP_NOT_ESTABLISHED"
    )
    full_result, full_tails = group_tolerance_qhat(
        development,
        development.absolute_error.to_numpy(float) / a4_scale,
        within_group_coverage=float(alg["within_group_coverage"]),
        target_group_success_probability=float(alg["target_group_success_probability"]),
    )
    if full_result.group_rank != 36:
        raise RuntimeError("Expected full-Development maximum-tail rank 36")
    full_tails.to_csv(out / "full_development_group_tails.csv", index=False)
    theory = {
        "outer_lobo": {
            "n_groups": 35,
            "group_rank": 35,
            "guaranteed_group_success_probability": 35 / 36,
        },
        "full_development": full_result.to_dict(),
        "scope": guarantee_scope(),
    }
    (out / "theoretical_guarantee.json").write_text(
        json.dumps(theory, indent=2), encoding="utf-8"
    )
    worst_row = pmetrics.loc[pmetrics.picp.idxmin()]
    selection = {
        "stage": cfg["stage"],
        "verdict": verdict,
        "adopted": established,
        "development_groups": 36,
        "heldout_groups_picp_at_least_0_90": groups_ge90,
        "median_normalized_width_ratio_vs_a4": width_ratio,
        "worst_background": str(worst_row.heldout_background),
        "worst_background_picp": float(worst_row.picp),
        "gates": gates,
        "confirmatory_test_used": False,
        "formal_qhat_fitted": False,
        "hpr_retrained": False,
        "further_hcp_variant_search_allowed_if_failed": False,
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if established:
        artifact = {
            "algorithm": alg["name"],
            "protocol_version": cfg["protocol_version"],
            "within_group_coverage": alg["within_group_coverage"],
            "target_group_success_probability": alg["target_group_success_probability"],
            "development_group_rank": full_result.group_rank,
            "development_rank_guarantee": full_result.guaranteed_group_success_probability,
            "frozen_a4_artifact": base["artifact"],
            "frozen_a4_artifact_sha256": base["artifact_sha256"],
            "formal_qhat_fitted": False,
            "future_independent_calibration_required": True,
            "future_independent_confirmatory_test_required": True,
            "confirmatory_test_used": False,
            "config_sha256": hashes["config_sha256"],
        }
        (out / "bti_worst_group_guard_hcp_frozen.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )

    lines = [
        "# BTI-WorstGroupGuard-HCP Development",
        "",
        f"- Verdict: **{verdict}**",
        "- Fixed 0.95 group-tail target; each 35-group LOBO training fold uses the maximum training group 90% tail.",
        "- 36-background outer LOBO; Confirmatory Test access=0; formal q_hat not fitted.",
        "- The rank guarantee assumes exchangeable group-tail thresholds and a shared Level-B protocol.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.method}: median PICP={row.background_median_picp:.6f}, "
            f"N<.85={int(row.n_picp_lt_0_85)}, N<.80={int(row.n_picp_lt_0_80)}, "
            f"MPIW={row.background_median_mpiw:.6f}, "
            f"normalized width={row.background_median_normalized_width:.6f}"
        )
    lines += [
        "",
        f"- heldout backgrounds PICP>=0.90: {groups_ge90}/36",
        f"- worst background: {worst_row.heldout_background}",
        f"- worst-background PICP: {float(worst_row.picp):.6f}",
        f"- normalized-width ratio vs A4: {width_ratio:.6f}",
        f"- LOBO rank guarantee: 35/36={35/36:.6f}",
        f"- full-Development rank guarantee: 36/37={36/37:.6f}",
        f"- gates={json.dumps(gates, ensure_ascii=False)}",
    ]
    (out / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
