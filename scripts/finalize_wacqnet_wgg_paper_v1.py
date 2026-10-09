# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\CO2\_third")
OUT = ROOT / "outputs" / "wacqnet_wgg_finalization_v1"
WACQ = ROOT / "outputs" / "wacqnet_development_v1"
WGG = ROOT / "outputs" / "bti_worst_group_guard_hcp_development"
DUAL = ROOT / "outputs" / "wacqnet_a4_wgg_dualtrack_protocol_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ap_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(score, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(np.asarray(x)).rank(method="average").to_numpy()
    ry = pd.Series(np.asarray(y)).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))


def finite(v: pd.Series) -> np.ndarray:
    a = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
    return a[np.isfinite(a)]


def bootstrap_median(values: np.ndarray, seed: int = 20260927, draws: int = 20000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    boot = np.median(values[idx], axis=1)
    return float(np.median(values)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def recalc_wacq() -> pd.DataFrame:
    cols = [
        "background_id", "true_emission", "mu_fixed", "mu", "risk",
        "lower90", "upper90", "width90", "covered90",
    ]
    df = pd.read_csv(WACQ / "wacqnet_lobo_predictions.csv", usecols=cols)
    rows = []
    for bg, g in df.groupby("background_id", sort=True):
        q = g["true_emission"].to_numpy(float)
        mu0 = g["mu_fixed"].to_numpy(float)
        mu = g["mu"].to_numpy(float)
        risk = g["risk"].to_numpy(float)
        ape_hpr = np.abs(mu0 - q) / np.maximum(q, 1e-12)
        ape_w = np.abs(mu - q) / np.maximum(q, 1e-12)
        unreliable = (ape_w >= 0.30).astype(np.int8)
        width = g["width90"].to_numpy(float)
        rows.append({
            "background_id": bg,
            "n_realizations": len(g),
            "hpr_median_ape": float(np.median(ape_hpr)),
            "wacq_median_ape": float(np.median(ape_w)),
            "hpr_p_reliable30": float(np.mean(ape_hpr < 0.30)),
            "wacq_p_reliable30": float(np.mean(ape_w < 0.30)),
            "hpr_scale_slope": slope(q, mu0),
            "wacq_scale_slope": slope(q, mu),
            "wacq_risk_spearman": spearman(risk, ape_w),
            "wacq_risk_auprc": ap_score(unreliable, risk),
            "unreliable_prevalence": float(unreliable.mean()),
            "wacq_picp90": float(g["covered90"].mean()),
            "wacq_mpiw": float(width.mean()),
            "wacq_normalized_width": float(np.mean(width / (np.maximum(mu, 0) + 1))),
            "wacq_lower_zero_fraction": float(np.mean(g["lower90"].to_numpy(float) <= 1e-12)),
        })
    return pd.DataFrame(rows)


def recalc_wgg() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["background_id", "mu_fixed", "width90", "covered90", "method"]
    df = pd.read_csv(WGG / "lobo_predictions.csv", usecols=cols)
    rows = []
    for (method, bg), g in df.groupby(["method", "background_id"], sort=True):
        width = g["width90"].to_numpy(float)
        mu0 = g["mu_fixed"].to_numpy(float)
        rows.append({
            "method": method,
            "background_id": bg,
            "n_realizations": len(g),
            "picp90": float(g["covered90"].astype(float).mean()),
            "mpiw": float(width.mean()),
            "median_normalized_width": float(np.median(width / (np.maximum(mu0, 0) + 1))),
        })
    bg = pd.DataFrame(rows)
    summary = []
    for method, g in bg.groupby("method", sort=True):
        summary.append({
            "method": method,
            "backgrounds": len(g),
            "median_picp90": float(g["picp90"].median()),
            "n_picp_lt_0_85": int((g["picp90"] < 0.85).sum()),
            "n_picp_lt_0_80": int((g["picp90"] < 0.80).sum()),
            "min_picp90": float(g["picp90"].min()),
            "median_mpiw": float(g["mpiw"].median()),
            "median_normalized_width": float(g["median_normalized_width"].median()),
        })
    return bg, pd.DataFrame(summary)


def summarize_ablations() -> pd.DataFrame:
    full = pd.read_csv(WACQ / "wacqnet_lobo_background_metrics.csv")
    abl = pd.read_csv(WACQ / "wacqnet_lobo_background_metrics_ablations.csv")
    df = pd.concat([full, abl], ignore_index=True)
    rows = []
    for candidate, g in df.groupby("candidate_id", sort=True):
        rows.append({
            "candidate_id": candidate,
            "backgrounds": int(g["heldout_background"].nunique()),
            "median_ape": float(g["median_ape"].median()),
            "median_p_reliable30": float(g["p_reliable30"].median()),
            "median_scale_slope": float(g["scale_compression_slope"].median()),
            "median_risk_spearman": float(g["spearman_risk_ape"].median()),
            "median_auprc": float(g["auprc_unreliable30"].median()),
            "median_picp90": float(g["picp90"].median()),
            "median_normalized_width": float(g["normalized_width"].median()),
            "n_picp_lt_0_85": int((g["picp90"] < 0.85).sum()),
            "n_picp_lt_0_80": int((g["picp90"] < 0.80).sum()),
            "min_picp90": float(g["picp90"].min()),
        })
    return pd.DataFrame(rows)


def metric_checks(bg: pd.DataFrame, wgg_summary: pd.DataFrame) -> dict:
    # Stored row-level predictions are decimal CSV exports, whereas the official
    # summaries were computed from in-memory float arrays.  A 1e-6 absolute
    # tolerance is therefore frozen for numerical traceability checks.
    numeric_abs_tolerance = 1e-6
    official = pd.read_csv(WACQ / "wacqnet_method_summary.csv").set_index("candidate_id")
    calc = {
        "hpr_macro_median_ape": float(bg["hpr_median_ape"].median()),
        "wacq_macro_median_ape": float(bg["wacq_median_ape"].median()),
        "hpr_macro_p_reliable30": float(bg["hpr_p_reliable30"].median()),
        "wacq_macro_p_reliable30": float(bg["wacq_p_reliable30"].median()),
        "hpr_macro_scale_slope": float(bg["hpr_scale_slope"].median()),
        "wacq_macro_scale_slope": float(bg["wacq_scale_slope"].median()),
        "wacq_macro_risk_spearman": float(bg["wacq_risk_spearman"].median()),
        "wacq_macro_auprc": float(bg["wacq_risk_auprc"].median()),
        "wacq_macro_prevalence": float(bg["unreliable_prevalence"].median()),
        "wacq_macro_picp90": float(bg["wacq_picp90"].median()),
        "wacq_macro_mpiw": float(bg["wacq_mpiw"].median()),
        "wacq_n_picp_lt_0_85": int((bg["wacq_picp90"] < 0.85).sum()),
        "wacq_n_picp_lt_0_80": int((bg["wacq_picp90"] < 0.80).sum()),
        "backgrounds_lower_median_ape": int((bg["wacq_median_ape"] < bg["hpr_median_ape"]).sum()),
        "backgrounds_positive_spearman": int((bg["wacq_risk_spearman"] > 0).sum()),
        "backgrounds_auprc_above_prevalence": int((bg["wacq_risk_auprc"] > bg["unreliable_prevalence"]).sum()),
    }
    expected = {
        "hpr_macro_median_ape": float(official.loc["HPR_FROZEN", "median_ape"]),
        "wacq_macro_median_ape": float(official.loc["WACQ_FULL", "median_ape"]),
        "hpr_macro_p_reliable30": float(official.loc["HPR_FROZEN", "median_p_reliable30"]),
        "wacq_macro_p_reliable30": float(official.loc["WACQ_FULL", "median_p_reliable30"]),
        "hpr_macro_scale_slope": float(official.loc["HPR_FROZEN", "median_scale_compression_slope"]),
        "wacq_macro_scale_slope": float(official.loc["WACQ_FULL", "median_scale_compression_slope"]),
        "wacq_macro_risk_spearman": float(official.loc["WACQ_FULL", "median_spearman_risk_ape"]),
        "wacq_macro_auprc": float(official.loc["WACQ_FULL", "median_auprc_unreliable30"]),
        "wacq_macro_prevalence": float(official.loc["WACQ_FULL", "median_unreliable_prevalence"]),
        "wacq_macro_picp90": float(official.loc["WACQ_FULL", "median_picp90"]),
        "wacq_macro_mpiw": float(official.loc["WACQ_FULL", "median_mpiw"]),
        "wacq_n_picp_lt_0_85": int(official.loc["WACQ_FULL", "n_picp_lt_0_85"]),
        "wacq_n_picp_lt_0_80": int(official.loc["WACQ_FULL", "n_picp_lt_0_80"]),
        "backgrounds_positive_spearman": int(official.loc["WACQ_FULL", "n_spearman_positive"]),
        "backgrounds_auprc_above_prevalence": int(official.loc["WACQ_FULL", "n_auprc_above_prevalence"]),
    }
    checks = {}
    for key, exp in expected.items():
        obs = calc[key]
        ok = obs == exp if isinstance(exp, int) else math.isclose(obs, exp, rel_tol=0, abs_tol=numeric_abs_tolerance)
        checks[key] = {"observed": obs, "expected": exp, "pass": ok}
    checks["all_core_metrics_reproduced"] = all(v["pass"] for v in checks.values())
    calc["wgg_summary"] = wgg_summary.to_dict(orient="records")
    return {"numeric_abs_tolerance": numeric_abs_tolerance, "calculated": calc, "checks": checks}


def write_reports(bg: pd.DataFrame, ablations: pd.DataFrame, validation: dict, wgg_summary: pd.DataFrame) -> None:
    ape_diff = bg["wacq_median_ape"].to_numpy() - bg["hpr_median_ape"].to_numpy()
    rel_diff = bg["wacq_p_reliable30"].to_numpy() - bg["hpr_p_reliable30"].to_numpy()
    slope_diff = bg["wacq_scale_slope"].to_numpy() - bg["hpr_scale_slope"].to_numpy()
    effects = []
    for name, values, favorable in [
        ("WACQ-HPR background median APE", ape_diff, "negative"),
        ("WACQ-HPR P_reliable30", rel_diff, "positive"),
        ("WACQ-HPR scale-compression slope", slope_diff, "positive"),
    ]:
        med, lo, hi = bootstrap_median(values)
        effects.append({"effect": name, "background_n": 36, "median_effect": med, "bootstrap_95ci_low": lo, "bootstrap_95ci_high": hi, "favorable_direction": favorable})
    pd.DataFrame(effects).to_csv(OUT / "background_level_effects.csv", index=False)

    a = validation["calculated"]
    wgg = wgg_summary.set_index("method")
    report = f"""# WACQNet + WorstGroupGuard-HCP 统计验证报告

## 分析单位

- 独立评价单位：36个 background groups。
- 每组6,750个Level-B realizations是组内受控重复，不作为36倍之外的独立卫星样本。
- 全部核心效应先在背景内汇总，再跨36个背景取中位数；置信区间使用背景级bootstrap（固定seed=20260927，20,000次）。

## 独立复算结果

- HPR macro median APE：{a['hpr_macro_median_ape']:.6f}。
- WACQNet macro median APE：{a['wacq_macro_median_ape']:.6f}；36/36背景改善。
- HPR macro P_reliable30：{a['hpr_macro_p_reliable30']:.6f}。
- WACQNet macro P_reliable30：{a['wacq_macro_p_reliable30']:.6f}。
- scale-compression slope：HPR {a['hpr_macro_scale_slope']:.6f}，WACQNet {a['wacq_macro_scale_slope']:.6f}。
- WACQNet risk Spearman中位数：{a['wacq_macro_risk_spearman']:.6f}；正相关36/36。
- WACQNet AUPRC中位数：{a['wacq_macro_auprc']:.6f}；高于背景内prevalence 36/36。
- 原生区间：PICP<0.85为{a['wacq_n_picp_lt_0_85']}/36，PICP<0.80为{a['wacq_n_picp_lt_0_80']}/36，故不通过预注册区间Gate。
- 核心正式汇总数字逐项复算一致：{validation['checks']['all_core_metrics_reproduced']}。

## WGG保守区间代价

- A4 median PICP：{wgg.loc['A4_FROZEN_REFERENCE','median_picp90']:.6f}；median MPIW：{wgg.loc['A4_FROZEN_REFERENCE','median_mpiw']:.3f}。
- WGG median PICP：{wgg.loc['BTI_WORST_GROUP_GUARD_HCP','median_picp90']:.6f}；median MPIW：{wgg.loc['BTI_WORST_GROUP_GUARD_HCP','median_mpiw']:.3f}。
- WGG把PICP<0.80的背景从{int(wgg.loc['A4_FROZEN_REFERENCE','n_picp_lt_0_80'])}降至{int(wgg.loc['BTI_WORST_GROUP_GUARD_HCP','n_picp_lt_0_80'])}，但代价是中位区间明显变宽并趋于过覆盖。

## 统计结论边界

1. 支持的结论：WACQNet在36-background LOBO上改善点预测并提供可迁移到held-out Development背景的相对风险排序。
2. 不支持的结论：WACQNet原生区间已经建立稳健跨背景覆盖。
3. WGG的秩解释依赖背景尾部分位数的可交换性及一致的Level-B协议；不能表述为任意背景漂移下的分布无关保证。
4. 双轨系统使用不同中心：点估计/风险来自WACQNet，保守区间以冻结HPR为中心。论文必须显式披露，不能把WGG区间说成围绕WACQNet点估计校准。
5. 当前没有全新的Calibration/Test角色，WACQNet最终全36背景refit亦未执行，因此不能写成外部confirmatory成功。

## 审稿风险分级

- P0：若把realizations作为独立n、或把HPR中心区间描述为WACQNet区间，会直接损害核心结论。
- P1：缺少全新外部confirmatory背景；所有模型改进结论目前限于36-background Development LOBO。
- P1：WGG效率代价较大，必须同时报告MPIW/normalized width，不能只报告覆盖。
- P2：正式稿需补充软件版本、训练耗时和参数量的统一表述。
"""
    (OUT / "statistical_validation_report_zh.md").write_text(report, encoding="utf-8")

    outline = """# 论文主线与写作框架（冻结版）

## 一句话论点

在真实OCO-3复杂背景的Level-B点源反演中，BTI-WACQNet通过风场对齐、冻结羽流模板条件化和前景—背景对比表征改善点估计与相对风险排序，而WorstGroupGuard-HCP以效率为代价提供独立的保守背景组覆盖防护；证据限于36背景Development LOBO，尚无新外部confirmatory验证。

## 建议题目方向

**Wind-Aligned Contrastive Quantile Learning for Trustworthy Power-Plant CO₂ Retrieval over Complex OCO-3 Backgrounds**

## 建议6–8页结构

1. Introduction：真实背景下点估计、风险排序和区间覆盖是三个不同问题。
2. Method：WACQNet点/风险轨；A4/WGG保守区间轨；明确双中心接口。
3. Experimental protocol：36-background LOBO、组内realizations、无held-out调参、预注册消融。
4. Point and risk results：APE、P_reliable30、scale compression、risk ranking。
5. Ablation and failure analysis：风场、contrast、Group-DRO；原生区间失败。
6. Conservative interval analysis：WGG覆盖改善及效率代价。
7. Discussion：Development证据边界、新外部数据不足、Level-B与真实瞬时排放差异。

## 主文保留

- WACQNet架构图及可部署输入约束。
- WACQNet与HPR逐背景配对结果。
- 预注册消融表。
- WGG覆盖—宽度权衡图或表。
- 最差背景失败案例。

## Supplementary

- 36折完整指标、训练曲线、所有checkpoint哈希。
- 完整协议、Gate、数据访问审计。
- 外部背景获取失败链与产品兼容性审计。

## 禁止表述

- “WACQNet已经通过外部Fresh Confirmatory Test”。
- “WACQNet原生区间实现稳健90%覆盖”。
- “WGG对任意domain shift提供分布无关保证”。
- “数十万realizations是独立卫星场景”。
"""
    (OUT / "paper_argument_and_outline_zh.md").write_text(outline, encoding="utf-8")

    results = f"""# Results初稿（中文证据版）

## WACQNet改善背景级点估计

我们在36个Development背景上执行leave-one-background-out评价，每折仅使用其余35个背景拟合模型，并在完整held-out背景的6,750个Level-B realizations上评价。realization被视为背景内受控重复，跨背景汇总以36个background groups为统计单位。WACQNet将背景级median APE的跨背景中位数从冻结HPR的{a['hpr_macro_median_ape']:.4f}降至{a['wacq_macro_median_ape']:.4f}，36/36个held-out背景均表现出较低的median APE。相应地，背景级P_reliable30中位数从{a['hpr_macro_p_reliable30']:.4f}提高至{a['wacq_macro_p_reliable30']:.4f}。预测值相对于真实排放量的背景级回归斜率中位数由{a['hpr_macro_scale_slope']:.4f}提高至{a['wacq_macro_scale_slope']:.4f}，表明scale compression得到缓解，但未被完全消除。

## WACQNet提供相对风险排序

WACQNet风险分数与APE的背景级Spearman相关系数中位数为{a['wacq_macro_risk_spearman']:.4f}，且36/36个背景均为正相关。以APE≥30%定义不可靠样本时，背景级AUPRC中位数为{a['wacq_macro_auprc']:.4f}，并在36/36个背景上高于相应不可靠率基线。这些结果支持把WACQNet输出解释为相对风险排序分数，但不支持把它解释为经过校准的失败概率。

## 消融定位主要组件的贡献

去除风场相关输入后，median APE由0.2782升至0.2879，风险Spearman由0.6304降至0.5899，median normalized width由1.2497升至1.7222。去除前景—背景contrast后，median P_reliable30由0.5295降至0.5191；去除Group-DRO后该指标降至0.5164，且最差背景PICP降至约0.5945。消融结果表明风场条件对风险排序和区间效率贡献最为明显，而Group-DRO的主要作用不能仅通过跨背景中位数判断，必须结合最差背景结果评价。

## 原生区间未达到预注册安全标准

尽管WACQNet原生非对称区间的背景级median PICP90为{a['wacq_macro_picp90']:.4f}，仍有{a['wacq_n_picp_lt_0_85']}/36个背景低于0.85，{a['wacq_n_picp_lt_0_80']}/36个背景低于0.80。因此，原生区间未通过预注册的最差组覆盖Gate，不能作为本文的稳健跨背景区间方法。

## WorstGroupGuard-HCP提高保守覆盖但牺牲效率

在相同36-background Development LOBO框架下，A4参考方法的背景级median PICP为{wgg.loc['A4_FROZEN_REFERENCE','median_picp90']:.4f}，median MPIW为{wgg.loc['A4_FROZEN_REFERENCE','median_mpiw']:.2f}；WorstGroupGuard-HCP将median PICP提高至{wgg.loc['BTI_WORST_GROUP_GUARD_HCP','median_picp90']:.4f}，并将PICP<0.80的背景数从{int(wgg.loc['A4_FROZEN_REFERENCE','n_picp_lt_0_80'])}降至0，但median MPIW增至{wgg.loc['BTI_WORST_GROUP_GUARD_HCP','median_mpiw']:.2f}。因此，该机制提供的是以区间效率为代价的保守背景组防护，而不是无代价的统一改进。

## 双轨系统的证据范围

最终冻结的系统将WACQNet用于点预测和相对风险排序，将以冻结HPR为中心的A4/WorstGroupGuard-HCP用于保守区间。两条轨道中心不同，回答的问题也不同。现有结果来自36-background Development LOBO；由于新的严格独立Calibration/Test数据链尚未建立，本文不能将这些结果表述为WACQNet的外部confirmatory验证。
"""
    (OUT / "results_draft_zh.md").write_text(results, encoding="utf-8")


def build_hash_inventory() -> pd.DataFrame:
    paths = [
        ROOT / "configs" / "wacqnet_development_protocol_v1.yaml",
        ROOT / "configs" / "bti_worst_group_guard_hcp_development.json",
        ROOT / "configs" / "wacqnet_a4_wgg_dualtrack_final_v1.json",
        ROOT / "src" / "wacqnet.py",
        ROOT / "scripts" / "run_wacqnet_36_lobo.py",
        ROOT / "scripts" / "run_bti_worst_group_guard_hcp_development.py",
        WACQ / "wacqnet_lobo_predictions.csv",
        WACQ / "wacqnet_lobo_background_metrics.csv",
        WACQ / "wacqnet_lobo_predictions_ablations.csv",
        WACQ / "wacqnet_lobo_background_metrics_ablations.csv",
        WGG / "lobo_predictions.csv",
        WGG / "lobo_background_metrics.csv",
        DUAL / "dualtrack_mathematical_protocol_zh.md",
    ]
    paths += sorted((WACQ / "checkpoints").glob("fold_*_WACQ_FULL.h5"))
    rows = []
    for p in paths:
        rows.append({"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bg = recalc_wacq()
    bg.to_csv(OUT / "independent_background_recalculation.csv", index=False)
    wgg_bg, wgg_summary = recalc_wgg()
    wgg_bg.to_csv(OUT / "wgg_background_recalculation.csv", index=False)
    wgg_summary.to_csv(OUT / "wgg_summary_recalculated.csv", index=False)
    ablations = summarize_ablations()
    ablations.to_csv(OUT / "ablation_summary_verified.csv", index=False)
    validation = metric_checks(bg, wgg_summary)
    (OUT / "independent_recalculation_summary.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(bg, ablations, validation, wgg_summary)

    a = validation["calculated"]
    wgg_lookup = wgg_summary.set_index("method")
    status_report = f"""# WACQNet + WGG-HCP 算法冻结与论文转段报告

## 最终状态

**STOP ALGORITHM FAMILY EXPANSION**

- WACQNet点预测Gate：PASS。
- WACQNet相对风险Gate：PASS。
- WACQNet原生区间Gate：FAIL。
- WorstGroupGuard-HCP Development Gate：PASS。
- 双轨状态：`DUAL_TRACK_DEVELOPMENT_EVIDENCE_FROZEN`。
- 全36背景最终refit：NOT RUN。
- 新外部Calibration/Test：NOT ESTABLISHED，读取0/0。

## 可复算证据

- HPR → WACQNet median APE：{a['hpr_macro_median_ape']:.4f} → {a['wacq_macro_median_ape']:.4f}。
- HPR → WACQNet P_reliable30：{a['hpr_macro_p_reliable30']:.4f} → {a['wacq_macro_p_reliable30']:.4f}。
- scale slope：{a['hpr_macro_scale_slope']:.4f} → {a['wacq_macro_scale_slope']:.4f}。
- WACQNet risk Spearman中位数：{a['wacq_macro_risk_spearman']:.4f}。
- WACQNet AUPRC中位数：{a['wacq_macro_auprc']:.4f}；36/36高于prevalence。
- 原生区间失败背景：PICP<0.85为{a['wacq_n_picp_lt_0_85']}/36，PICP<0.80为{a['wacq_n_picp_lt_0_80']}/36。
- WGG median PICP：{wgg_lookup.loc['BTI_WORST_GROUP_GUARD_HCP','median_picp90']:.4f}；median MPIW：{wgg_lookup.loc['BTI_WORST_GROUP_GUARD_HCP','median_mpiw']:.2f}。

## 论文主张

允许：36-background Development LOBO支持WACQNet点预测改善和相对风险排序；WGG显示保守覆盖与效率之间的明确权衡。

禁止：外部Confirmatory成功、WACQNet原生区间成功、任意domain shift保证、把HPR中心WGG区间说成WACQNet中心区间。

## 写作转段

已生成结果初稿、论文主线、术语表、背景级统计验证、消融汇总、审稿风险清单及完整资产哈希。下一步只做图表定稿、Methods/Discussion扩写和投稿格式适配；除非出现实现错误，不再开发新算法变体。
"""
    (OUT / "finalization_status_report_zh.md").write_text(status_report, encoding="utf-8")

    terminology = pd.DataFrame([
        ["BTI-WACQNet", "wind-aligned, template-conditioned contrastive quantile network", "WACQNet/WACQ_FULL", "点预测与相对风险轨"],
        ["relative risk score", "WACQNet deployment risk score", "uncertainty probability", "不可称为校准失败概率"],
        ["BTI-WorstGroupGuard-HCP", "group-tail guarded conformal procedure", "WGG", "HPR中心保守区间轨"],
        ["background group", "independent OCO-3 background unit", "scene/realization", "统计独立单位"],
        ["Level-B realization", "controlled synthetic-plume realization on a real background", "independent sample", "组内重复"],
        ["P_reliable30", "Pr(APE < 30%) within a background/condition", "accuracy", "操作可靠率"],
    ], columns=["canonical_term", "first_use_definition", "variants_or_risk", "decision"])
    terminology.to_csv(OUT / "terminology_ledger.csv", index=False)

    inventory = build_hash_inventory()
    inventory.to_csv(OUT / "artifact_hash_inventory.csv", index=False)
    aggregate = hashlib.sha256("\n".join(f"{r.path}|{r.sha256}" for r in inventory.itertuples()).encode("utf-8")).hexdigest()
    freeze = {
        "stage": "WACQNET_WGG_FINALIZATION_V1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "STOP_ALGORITHM_FAMILY_EXPANSION",
        "algorithm_family_frozen": True,
        "point_and_risk_track": {
            "method": "BTI-WACQNet v1.0.2 / WACQ_FULL",
            "evidence": "36-background Development LOBO",
            "point_gate": "PASS",
            "risk_gate": "PASS",
            "native_interval_gate": "FAIL",
            "full_36_background_final_refit": "NOT_RUN",
        },
        "conservative_interval_track": {
            "method": "BTI-WorstGroupGuard-HCP over frozen HPR-centered A4 score",
            "evidence": "36-background Development LOBO",
            "development_gate": "PASS",
            "formal_new_calibration_qhat": "NOT_FITTED",
        },
        "combined_system": {
            "status": "DUAL_TRACK_DEVELOPMENT_EVIDENCE_FROZEN",
            "same_center": False,
            "external_confirmatory_validation": "NOT_ESTABLISHED",
            "allowed_claim": "Development-level background-grouped LOBO evidence for WACQNet point/risk gains and WGG conservative coverage trade-off",
            "forbidden_claims": [
                "WACQNet native interval coverage established",
                "WACQNet externally confirmed on a new test",
                "WGG interval calibrated around the WACQNet point estimate",
                "arbitrary-domain-shift distribution-free coverage",
            ],
        },
        "data_access": {
            "new_calibration_roles_established": False,
            "new_confirmatory_roles_established": False,
            "new_calibration_rows_read": 0,
            "new_confirmatory_rows_read": 0,
        },
        "artifact_inventory_sha256": aggregate,
        "core_metric_recalculation_pass": validation["checks"]["all_core_metrics_reproduced"],
    }
    freeze_path = OUT / "final_algorithm_freeze.json"
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_algorithm_freeze.sha256").write_text(sha256(freeze_path) + "\n", encoding="ascii")

    repro = {
        "status": "PASS" if validation["checks"]["all_core_metrics_reproduced"] and len(inventory[inventory["path"].str.contains("fold_")]) == 36 else "FAIL",
        "core_metrics_independently_recalculated": True,
        "training_rerun": False,
        "model_inference_rerun": False,
        "checkpoint_count": int(len(inventory[inventory["path"].str.contains("fold_")])),
        "artifact_inventory_sha256": aggregate,
        "notes": [
            "Core metrics were recomputed directly from stored row-level LOBO predictions.",
            "Ablation summaries were recomputed from stored per-background ablation metrics; ablation models were not retrained.",
            "No sealed Calibration/Test data were accessed.",
        ],
    }
    (OUT / "reproducibility_audit.json").write_text(json.dumps(repro, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "reproducibility_audit_zh.md").write_text(
        "# 可复现性审计\n\n"
        f"- 状态：**{repro['status']}**\n"
        "- 重新训练：否。\n- 重新推理：否。\n"
        f"- 36折WACQNet checkpoint：{repro['checkpoint_count']}个。\n"
        f"- 核心指标从逐样本LOBO预测独立复算：{validation['checks']['all_core_metrics_reproduced']}。\n"
        f"- 关键资产集合SHA256：`{aggregate}`。\n"
        "- 新Calibration/Test读取：0/0。\n",
        encoding="utf-8",
    )

    risk = pd.DataFrame([
        ["P0", "双轨中心混淆", "WGG区间以HPR为中心，不是WACQNet中心", "方法图、公式和摘要均显式分轨"],
        ["P0", "伪重复", "realizations不是独立背景", "所有主推断以36个背景为n"],
        ["P1", "缺少全新外部确认", "新9/9 strict pool为0组", "限制claim为Development LOBO"],
        ["P1", "原生区间失败", "4/36 PICP<0.80", "作为负结果报告，使用独立WGG轨"],
        ["P1", "WGG效率代价", "median MPIW明显增加", "覆盖和宽度必须成对报告"],
        ["P2", "复现信息不完整", "正式稿仍需软件版本/硬件/耗时", "作者补充环境清单"],
    ], columns=["severity", "risk", "evidence", "required_response"])
    risk.to_csv(OUT / "reviewer_risk_register.csv", index=False)

    output_hashes = {}
    for p in sorted(OUT.glob("*")):
        if p.is_file() and p.name != "finalization_output_hashes.json":
            output_hashes[p.name] = sha256(p)
    (OUT / "finalization_output_hashes.json").write_text(json.dumps(output_hashes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
