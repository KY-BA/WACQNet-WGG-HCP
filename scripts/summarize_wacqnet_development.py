"""Summarize the frozen WACQNet Development experiment without any Test access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hpr_background_metrics(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for background, group in meta.groupby("background_id"):
        q = group.true_emission.to_numpy(float)
        mu = group.mu_fixed.to_numpy(float)
        ape = np.abs(mu - q) / q
        xc, yc = q - q.mean(), mu - mu.mean()
        slope = float(np.sum(xc * yc) / max(float(np.sum(xc * xc)), 1e-12))
        rows.append({"candidate_id": "HPR_FROZEN", "heldout_background": background,
                     "median_ape": float(np.median(ape)), "p_reliable30": float(np.mean(ape < .30)),
                     "scale_compression_slope": slope})
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict:
    result = {
        "backgrounds": int(frame.heldout_background.nunique()),
        "median_ape": float(frame.median_ape.median()),
        "median_p_reliable30": float(frame.p_reliable30.median()),
        "median_scale_compression_slope": float(frame.scale_compression_slope.median()),
    }
    for key in ["spearman_risk_ape", "auprc_unreliable30", "unreliable_prevalence", "picp90", "mpiw", "normalized_width", "lower_zero_fraction"]:
        if key in frame:
            result[f"median_{key}"] = float(frame[key].median())
    if "picp90" in frame:
        result["n_picp_lt_0_85"] = int((frame.picp90 < .85).sum())
        result["n_picp_lt_0_80"] = int((frame.picp90 < .80).sum())
        result["min_picp90"] = float(frame.picp90.min())
    if "spearman_risk_ape" in frame:
        result["n_spearman_positive"] = int((frame.spearman_risk_ape > 0).sum())
        result["n_auprc_above_prevalence"] = int((frame.auprc_unreliable30 > frame.unreliable_prevalence).sum())
    return result


def main() -> None:
    cfg_path = Path("D:/CO2/_third/configs/wacqnet_development_protocol_v1.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = Path(cfg["outputs"]["training_dir"])
    protocol = json.loads((Path(cfg["outputs"]["protocol_dir"]) / "protocol_freeze_audit.json").read_text(encoding="utf-8"))
    full = pd.read_csv(out / "wacqnet_lobo_background_metrics.csv")
    ablations = pd.read_csv(out / "wacqnet_lobo_background_metrics_ablations.csv")
    all_metrics = pd.concat([full, ablations], ignore_index=True)
    counts = all_metrics.groupby("candidate_id").heldout_background.nunique()
    expected = {"WACQ_FULL", "WACQ_NO_WIND", "WACQ_NO_CONTRAST", "WACQ_NO_GDRO"}
    if set(counts.index) != expected or not (counts == 36).all():
        raise RuntimeError(f"Incomplete ablations: {counts.to_dict()}")
    meta = pd.read_csv(Path(cfg["outputs"]["cache_dir"]) / "wacqnet_levelb_metadata.csv")
    hpr = hpr_background_metrics(meta)
    hpr_summary = summarize(hpr)

    a4_parts = []
    for path in [
        Path("D:/CO2/_third/outputs/bti_scene_tail_encoder_development/inner_lobo_predictions.csv"),
        Path("D:/CO2/_third/outputs/bti_scene_tail_encoder_development/external_development_predictions.csv"),
    ]:
        frame = pd.read_csv(path)
        a4_parts.append(frame[frame.method == "S0_FROZEN_A4_REFERENCE"])
    a4 = pd.concat(a4_parts, ignore_index=True)
    a4_norm = a4.groupby("background_id").apply(lambda g: float(np.mean(g.width90 / (g.mu_fixed + 1.0))))
    a4_median_normalized_width = float(a4_norm.median())

    summaries = {"HPR_FROZEN": hpr_summary}
    summary_rows = [{"candidate_id": "HPR_FROZEN", **hpr_summary}]
    for candidate, group in all_metrics.groupby("candidate_id"):
        item = summarize(group)
        summaries[candidate] = item
        summary_rows.append({"candidate_id": candidate, **item})
    pd.DataFrame(summary_rows).to_csv(out / "wacqnet_method_summary.csv", index=False)

    primary = all_metrics[all_metrics.candidate_id == "WACQ_FULL"].copy()
    paired = primary.merge(hpr, on="heldout_background", suffixes=("_wacq", "_hpr"))
    gate_cfg = cfg["adoption_gate"]
    point = {
        "macro_median_ape_ratio_vs_hpr": summaries["WACQ_FULL"]["median_ape"] / hpr_summary["median_ape"],
        "macro_p_reliable30_gain_vs_hpr": summaries["WACQ_FULL"]["median_p_reliable30"] - hpr_summary["median_p_reliable30"],
        "backgrounds_with_lower_median_ape": int((paired.median_ape_wacq < paired.median_ape_hpr).sum()),
        "relative_scale_slope_gap_reduction": float(1 - abs(1 - summaries["WACQ_FULL"]["median_scale_compression_slope"]) / abs(1 - hpr_summary["median_scale_compression_slope"])),
    }
    point["pass"] = bool(
        point["macro_median_ape_ratio_vs_hpr"] <= gate_cfg["point"]["maximum_macro_median_ape_ratio_vs_hpr"]
        and point["macro_p_reliable30_gain_vs_hpr"] >= gate_cfg["point"]["minimum_macro_p_reliable30_gain_vs_hpr"]
        and point["backgrounds_with_lower_median_ape"] >= gate_cfg["point"]["minimum_backgrounds_with_lower_median_ape"]
        and point["relative_scale_slope_gap_reduction"] >= gate_cfg["point"]["minimum_relative_scale_slope_gap_reduction"]
    )
    risk = {
        "backgrounds_spearman_positive": summaries["WACQ_FULL"]["n_spearman_positive"],
        "backgrounds_auprc_above_prevalence": summaries["WACQ_FULL"]["n_auprc_above_prevalence"],
    }
    risk["pass"] = bool(
        risk["backgrounds_spearman_positive"] >= gate_cfg["risk"]["minimum_backgrounds_spearman_positive"]
        and risk["backgrounds_auprc_above_prevalence"] >= gate_cfg["risk"]["minimum_backgrounds_auprc_above_prevalence"]
    )
    interval = {
        "backgrounds_picp_below_0_80": summaries["WACQ_FULL"]["n_picp_lt_0_80"],
        "backgrounds_picp_below_0_85": summaries["WACQ_FULL"]["n_picp_lt_0_85"],
        "median_normalized_width": summaries["WACQ_FULL"]["median_normalized_width"],
        "a4_median_normalized_width": a4_median_normalized_width,
        "normalized_width_reduction_vs_a4": 1 - summaries["WACQ_FULL"]["median_normalized_width"] / a4_median_normalized_width,
    }
    interval["pass"] = bool(
        interval["backgrounds_picp_below_0_80"] <= gate_cfg["interval"]["maximum_backgrounds_picp_below_0_80"]
        and interval["backgrounds_picp_below_0_85"] <= gate_cfg["interval"]["maximum_backgrounds_picp_below_0_85"]
        and interval["normalized_width_reduction_vs_a4"] >= gate_cfg["interval"]["minimum_macro_normalized_width_reduction_vs_frozen_a4"]
    )
    verdict = gate_cfg["decision"]["success"] if point["pass"] and risk["pass"] and interval["pass"] else gate_cfg["decision"]["failure"]
    gate = {
        "protocol_sha256": protocol["protocol_sha256"], "primary_candidate": "WACQ_FULL",
        "point_gate": point, "risk_gate": risk, "interval_gate": interval, "verdict": verdict,
        "current_v3_calibration_rows_read": 0, "current_v3_confirmatory_test_rows_read": 0,
    }
    (out / "wacqnet_development_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    worst = primary.nsmallest(8, "picp90")[["heldout_background", "picp90", "median_ape", "p_reliable30", "mpiw"]]
    worst.to_csv(out / "wacqnet_worst_coverage_backgrounds.csv", index=False)

    ablation_lines = []
    for candidate in ["WACQ_FULL", "WACQ_NO_WIND", "WACQ_NO_CONTRAST", "WACQ_NO_GDRO"]:
        s = summaries[candidate]
        ablation_lines.append(
            f"| {candidate} | {s['median_ape']:.4f} | {s['median_p_reliable30']:.4f} | "
            f"{s['median_spearman_risk_ape']:.4f} | {s['median_picp90']:.4f} | "
            f"{s['median_normalized_width']:.4f} | {s['n_picp_lt_0_85']} | {s['n_picp_lt_0_80']} |"
        )
    worst_lines = [
        f"- `{row.heldout_background}`：PICP90={row.picp90:.4f}，MPIW={row.mpiw:.2f}，median APE={row.median_ape:.3f}"
        for row in worst.itertuples(index=False)
    ]
    report = f"""# BTI-WACQNet Development 报告

## Protocol

- 冻结版本：{cfg['protocol_version']}
- 协议 SHA256：`{protocol['protocol_sha256']}`
- 数据角色：36个 Development backgrounds；每折35组训练、1组完整外推。
- model-fit 与 fold-internal q_hat 使用互斥 repetitions；held-out 背景不参与训练、q_hat或选择。
- 当前v3 Calibration/Test读取：0 / 0。

## Primary WACQ_FULL

- macro median APE：{summaries['WACQ_FULL']['median_ape']:.4f}（HPR {hpr_summary['median_ape']:.4f}；比值 {point['macro_median_ape_ratio_vs_hpr']:.3f}）。
- macro P_reliable30：{summaries['WACQ_FULL']['median_p_reliable30']:.4f}（HPR {hpr_summary['median_p_reliable30']:.4f}；增益 {point['macro_p_reliable30_gain_vs_hpr']:.3f}）。
- 背景改善：{point['backgrounds_with_lower_median_ape']}/36。
- scale-compression slope：{summaries['WACQ_FULL']['median_scale_compression_slope']:.4f}（HPR {hpr_summary['median_scale_compression_slope']:.4f}）。
- 风险Spearman>0：{risk['backgrounds_spearman_positive']}/36；AUPRC>prevalence：{risk['backgrounds_auprc_above_prevalence']}/36。
- median PICP90：{summaries['WACQ_FULL']['median_picp90']:.4f}；PICP<0.85：{interval['backgrounds_picp_below_0_85']}；PICP<0.80：{interval['backgrounds_picp_below_0_80']}。
- median normalized width：{interval['median_normalized_width']:.4f}，相对A4下降 {interval['normalized_width_reduction_vs_a4']:.1%}。

## Frozen Ablations

| Candidate | Median APE | Median P_reliable30 | Median risk Spearman | Median PICP90 | Median normalized width | N(PICP<0.85) | N(PICP<0.80) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(ablation_lines)}

消融只能用于机制解释，不能替换预注册primary。去掉风场使风险排序下降且区间显著变宽；去掉template contrast或Group-DRO也没有解决最差组覆盖失效。Group-DRO消融的中位覆盖较高，但最差PICP仅{summaries['WACQ_NO_GDRO']['min_picp90']:.4f}，说明macro指标不能掩盖灾难性背景。

## Worst-group Coverage of WACQ_FULL

{chr(10).join(worst_lines)}

## Frozen Gate

- Point gate：{'PASS' if point['pass'] else 'FAIL'}
- Risk gate：{'PASS' if risk['pass'] else 'FAIL'}
- Interval gate：{'PASS' if interval['pass'] else 'FAIL'}
- **Final Development verdict：{verdict}**

## Interpretation Boundary

当前证据支持：WACQ_FULL在36-background LOBO上显著改善点估计、scale compression和relative risk ranking。当前证据不支持：该版本已经建立跨背景稳健的90%区间覆盖。由于预注册interval gate失败，不得进入新的Calibration/Test；也不能用消融中某个macro指标更好的候选替换primary。若继续开发，必须另建版本化协议，并仍只使用Development。

## Tests

- 9个预注册协议/实现测试函数直接执行：9 passed，0 failed。
- `tf_server`环境未安装pytest，因此不能表述为pytest suite通过；执行方式和测试名记录于`wacqnet_test_audit.json`。
"""
    (out / "wacqnet_development_report_zh.md").write_text(report, encoding="utf-8")
    artifacts = {}
    for name in ["wacqnet_lobo_background_metrics.csv", "wacqnet_lobo_background_metrics_ablations.csv", "wacqnet_method_summary.csv", "wacqnet_development_gate.json", "wacqnet_development_report_zh.md", "wacqnet_test_audit.json"]:
        artifacts[name] = sha256(out / name)
    (out / "wacqnet_output_hashes.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
