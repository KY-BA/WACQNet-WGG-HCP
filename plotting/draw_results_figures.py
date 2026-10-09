"""Read-only plotting of frozen Development summaries; no model execution.

Times New Roman and a 122 mm canvas are explicit user/template overrides.
Run with a Python environment containing numpy, pandas and matplotlib.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "text.color": "#26343D",
    "axes.labelcolor": "#26343D",
    "xtick.color": "#26343D",
    "ytick.color": "#26343D",
})

ROOT = Path(r"D:\CO2\_third")
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
DATA = OUT / "source_data"
SOURCE = ROOT / "outputs/wacqnet_wgg_finalization_v1"
width_mm = 122
png_dpi = 300
tiff_dpi = 600
N_GROUPS = 36
READ_ALLOWLIST = (
    "independent_background_recalculation.csv",
    "background_level_effects.csv",
    "wgg_background_recalculation.csv",
    "wgg_summary_recalculated.csv",
    "independent_recalculation_summary.json",
    "final_algorithm_freeze.json",
)
BLUE = "#145DA0"
PURPLE = "#7A6FAC"
ORANGE = "#C97723"
GREY = "#73808A"
INK = "#26343D"
RED = "#A64C49"
GRID = "#E4E8EC"
METHODS = ("WACQ_NATIVE", "A4_FROZEN_REFERENCE", "BTI_WORST_GROUP_GUARD_HCP")
LABELS = ("Native (WACQNet)", "A4 (HPR)", "WGG-HCP (HPR)")
COLORS = (BLUE, PURPLE, ORANGE)
MARKERS = ("o", "D", "s")
checks = []
artifacts = []


def require(condition, label):
    if not bool(condition):
        raise RuntimeError(label)
    checks.append({"check": label, "pass": True})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = Path("C:/Windows/Fonts") / name
        require(path.is_file(), f"font_exists:{name}")
        fm.fontManager.addfont(str(path))
    require(fm.FontProperties(fname="C:/Windows/Fonts/times.ttf").get_name()
            == "Times New Roman", "font_identity")
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)


def style(ax, grid=True):
    ax.tick_params(width=0.65, length=2.6, pad=2)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def title(ax, letter, text, subtitle=None):
    top = (1.27 if "\n" in subtitle else 1.20) if subtitle else 1.09
    ax.text(-0.02, top, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")
    ax.text(0.12, top, text, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom")
    if subtitle:
        # All numeric callouts live outside the data area so no points are hidden.
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=8,
                color=BLUE, va="bottom")


def export(fig, stem):
    fig.canvas.draw()
    # Keep the physical canvas width exactly 122 mm; do not auto-tight crop.
    renderer = fig.canvas.get_renderer()
    extents = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            extents.append((text.get_text(), box))
    frame = fig.bbox
    outside = [value for value, box in extents
               if box.x0 < -1 or box.y0 < -1 or box.x1 > frame.x1 + 1
               or box.y1 > frame.y1 + 1]
    require(not outside, f"all_text_inside_canvas:{stem}:{outside}")
    for suffix, options in (
        (".pdf", {}), (".svg", {}),
        (".png", {"dpi": png_dpi}),
        (".tiff", {"dpi": tiff_dpi, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = FIG / (stem + suffix)
        fig.savefig(path, **options)
        artifacts.append({"file": str(path.relative_to(OUT)),
                          "sha256": sha256(path)})
    plt.close(fig)


def load_and_check():
    before = {name: sha256(SOURCE / name) for name in READ_ALLOWLIST}
    bg = pd.read_csv(SOURCE / READ_ALLOWLIST[0])
    effects = pd.read_csv(SOURCE / READ_ALLOWLIST[1])
    wgg = pd.read_csv(SOURCE / READ_ALLOWLIST[2])
    official_wgg = pd.read_csv(SOURCE / READ_ALLOWLIST[3])
    expected = json.loads((SOURCE / READ_ALLOWLIST[4]).read_text(encoding="utf-8"))["calculated"]
    freeze = json.loads((SOURCE / READ_ALLOWLIST[5]).read_text(encoding="utf-8"))
    require(len(bg) == N_GROUPS and bg.background_id.nunique() == N_GROUPS,
            "all_36_unique_Development_groups")
    require((bg.n_realizations == 6750).all(), "6750_repeated_realizations_per_group")
    require(np.isfinite(bg.select_dtypes(include="number")).all().all(),
            "background_summary_has_no_nonfinite_values")
    require(set(wgg.method) == set(METHODS[1:]), "exact_A4_and_WGG_methods")
    require(not wgg.duplicated(["method", "background_id"]).any(), "unique_method_group_pairs")
    for method in METHODS[1:]:
        rows = wgg.loc[wgg.method == method]
        require(len(rows) == N_GROUPS and set(rows.background_id) == set(bg.background_id),
                f"same_36_groups:{method}")
        require((rows.n_realizations == 6750).all(), f"same_repeated_n:{method}")
    require(freeze["combined_system"]["same_center"] is False, "different_interval_centers_disclosed")
    require(freeze["combined_system"]["external_confirmatory_validation"] == "NOT_ESTABLISHED",
            "no_external_confirmatory_claim")
    expected_columns = {
        "hpr_median_ape": "hpr_macro_median_ape",
        "wacq_median_ape": "wacq_macro_median_ape",
        "hpr_p_reliable30": "hpr_macro_p_reliable30",
        "wacq_p_reliable30": "wacq_macro_p_reliable30",
        "hpr_scale_slope": "hpr_macro_scale_slope",
        "wacq_scale_slope": "wacq_macro_scale_slope",
        "wacq_risk_spearman": "wacq_macro_risk_spearman",
        "wacq_risk_auprc": "wacq_macro_auprc",
        "unreliable_prevalence": "wacq_macro_prevalence",
        "wacq_picp90": "wacq_macro_picp90",
        "wacq_mpiw": "wacq_macro_mpiw",
    }
    for column, key in expected_columns.items():
        require(abs(bg[column].median() - expected[key]) <= 1e-6, f"official_summary_matches:{key}")
    require((bg.wacq_median_ape < bg.hpr_median_ape).sum() == expected["backgrounds_lower_median_ape"],
            "APE_improvement_count_matches")
    require((bg.wacq_risk_spearman > 0).sum() == expected["backgrounds_positive_spearman"],
            "risk_positive_count_matches")
    require((bg.wacq_risk_auprc > bg.unreliable_prevalence).sum()
            == expected["backgrounds_auprc_above_prevalence"], "AP_above_prevalence_count_matches")
    # Read existing bootstrap estimates unchanged: no resampling or new fitting.
    effect_keys = ("WACQ-HPR background median APE", "WACQ-HPR P_reliable30",
                   "WACQ-HPR scale-compression slope")
    require(set(effects.effect) == set(effect_keys) and (effects.background_n == 36).all(),
            "existing_group_bootstrap_effects_only")
    for _, row in effects.iterrows():
        require(row.bootstrap_95ci_low <= row.median_effect <= row.bootstrap_95ci_high,
                f"existing_CI_order:{row.effect}")
    return bg, effects.set_index("effect").loc[list(effect_keys)].reset_index(), wgg, official_wgg, before


def point_figure(bg, effects):
    bg[["background_id", "n_realizations", "hpr_median_ape", "wacq_median_ape",
        "hpr_p_reliable30", "wacq_p_reliable30", "hpr_scale_slope", "wacq_scale_slope"]].to_csv(
            DATA / "figure_3_point_backgrounds.csv", index=False)
    effects.to_csv(DATA / "figure_3_existing_bootstrap_effects.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(width_mm / 25.4, 113 / 25.4))
    fig.subplots_adjust(left=0.145, right=0.965, bottom=0.12, top=0.83,
                        wspace=0.72, hspace=0.83)
    panels = (
        ("hpr_median_ape", "wacq_median_ape", "HPR median APE", "WACQNet median APE", "Point error", 2.7),
        ("hpr_p_reliable30", "wacq_p_reliable30", "HPR reliable fraction", "WACQNet reliable fraction", "Point reliability", 0.82),
        ("hpr_scale_slope", "wacq_scale_slope", "HPR OLS slope", "WACQNet OLS slope", "Scale compression", 1.09),
    )
    for ax, letter, (xcol, ycol, xl, yl, ttl, lim) in zip(axes.flat, "abc", panels):
        ax.plot([0, lim], [0, lim], color=GREY, linestyle="--", linewidth=0.8, zorder=1)
        ax.scatter(bg[xcol], bg[ycol], s=19, color=BLUE, alpha=0.85,
                   edgecolors="white", linewidth=0.35, zorder=3)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel=xl, ylabel=yl)
        ticks = {"a": [0, 1, 2], "b": [0, 0.2, 0.4, 0.6, 0.8], "c": [0, 0.5, 1]}[letter]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal", adjustable="box")
        if letter == "c":
            ax.axhline(1, color=GREY, linestyle=":", linewidth=0.85)
        annotation = f"Medians: {bg[xcol].median():.3f} → {bg[ycol].median():.3f}"
        style(ax)
        title(ax, letter, ttl, subtitle=annotation)
    ax = axes[1, 1]
    ys = np.array([2, 1, 0])
    for y, row in zip(ys, effects.itertuples(index=False)):
        ax.errorbar(row.median_effect, y,
                    xerr=[[row.median_effect-row.bootstrap_95ci_low],
                          [row.bootstrap_95ci_high-row.median_effect]],
                    fmt="o", color=BLUE, markersize=5, linewidth=1.4, capsize=3, zorder=3)
        ax.text(row.median_effect, y+0.23, f"{row.median_effect:+.3f}",
                ha="center", fontsize=8, color=BLUE)
    ax.axvline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax.set(xlim=(-0.37, 0.65), ylim=(-0.5, 2.6), xlabel="Paired-median difference")
    ax.set_yticks(ys, ["Median APE", "Reliable\nfraction", "OLS slope"])
    ax.set_xticks([-0.3, 0, 0.3, 0.6])
    style(ax)
    title(ax, "d", "Paired effects", subtitle="Existing 95% bootstrap CIs")
    fig.text(0.52, 0.964, "Point retrieval across 36 Development backgrounds",
             ha="center", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.018, "Dots: background summaries. Error bars: existing 95% group-bootstrap CIs.",
             ha="center", fontsize=8, color=GREY)
    export(fig, "figure_3_point_results")


def risk_figure(bg):
    risk = bg[["background_id", "n_realizations", "wacq_risk_spearman",
               "wacq_risk_auprc", "unreliable_prevalence"]].sort_values(
                   ["wacq_risk_spearman", "background_id"], kind="stable").copy()
    risk["display_index"] = np.arange(1, N_GROUPS+1)
    risk.to_csv(DATA / "figure_4_risk_backgrounds.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(width_mm / 25.4, 81 / 25.4))
    fig.subplots_adjust(left=0.14, right=0.965, top=0.715, bottom=0.24, wspace=0.48)
    ax = axes[0]
    ax.scatter(risk.display_index, risk.wacq_risk_spearman, s=14, color=BLUE,
               edgecolors="white", linewidth=0.3, zorder=3)
    median = risk.wacq_risk_spearman.median()
    ax.axhline(median, color=BLUE, linestyle="--", linewidth=0.9)
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set(xlim=(0, 37), ylim=(-0.025, 0.84), xlabel="Background display index",
           ylabel=r"Spearman $\rho$ (risk, APE)")
    ax.set_xticks([1, 12, 24, 36])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    style(ax)
    title(ax, "a", "Risk–error ranking", subtitle=f"36/36 positive\nMedian = {median:.3f}")
    ax = axes[1]
    ax.plot([0.3, 1], [0.3, 1], color=GREY, linestyle="--", linewidth=0.8)
    ax.scatter(risk.unreliable_prevalence, risk.wacq_risk_auprc, s=20,
               color=BLUE, edgecolors="white", linewidth=0.35, zorder=3)
    ax.set(xlim=(0.30, 1), ylim=(0.30, 1), xlabel="Failure prevalence",
           ylabel="Average precision (AUPRC)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.4, 0.6, 0.8, 1])
    ax.set_yticks([0.4, 0.6, 0.8, 1])
    style(ax)
    title(ax, "b", "Failure discrimination",
          subtitle=f"36/36 AP > prevalence\nMedian AP = {risk.wacq_risk_auprc.median():.3f}")
    fig.text(0.52, 0.945, "WACQNet relative-risk ranking: Development LOBO",
             ha="center", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.04, "Each dot: one background. Failure event: APE ≥ 0.30. Not calibrated probability.",
             ha="center", fontsize=8, color=GREY)
    export(fig, "figure_4_wacqnet_risk_ranking")


def prepare_intervals(bg, wgg, official_wgg):
    native = bg[["background_id", "n_realizations", "wacq_picp90", "wacq_mpiw"]].rename(
        columns={"wacq_picp90": "picp90", "wacq_mpiw": "mpiw"})
    native["method"] = METHODS[0]
    native["center"] = "WACQNet"
    others = wgg[["method", "background_id", "n_realizations", "picp90", "mpiw"]].copy()
    others["center"] = "Frozen HPR"
    all_rows = pd.concat([native, others], ignore_index=True)
    require(len(all_rows) == 108, "all_108_interval_points_retained")
    require(np.isfinite(all_rows[["picp90", "mpiw"]]).all().all(), "finite_interval_metrics")
    require(all_rows.picp90.between(0, 1).all() and (all_rows.mpiw > 0).all(),
            "valid_probability_and_width_ranges")
    order = native.sort_values(["picp90", "background_id"], kind="stable").background_id.tolist()
    ranks = {value: i+1 for i, value in enumerate(order)}
    all_rows["display_index_by_native_picp"] = all_rows.background_id.map(ranks)
    all_rows["efficiency_metric"] = "within_background_mean_interval_width"
    all_rows["width_unit"] = "Mt CO2 yr^-1"
    all_rows["evidence_role"] = "Development"
    all_rows.to_csv(DATA / "figure_5_interval_backgrounds.csv", index=False)
    summaries = []
    for method in METHODS:
        rows = all_rows.loc[all_rows.method == method]
        summary = {"method": method, "backgrounds": len(rows),
                   "median_picp90": rows.picp90.median(),
                   "min_picp90": rows.picp90.min(),
                   "n_picp_lt_0_85": int((rows.picp90 < 0.85).sum()),
                   "n_picp_lt_0_80": int((rows.picp90 < 0.80).sum()),
                   "median_mpiw": rows.mpiw.median(),
                   "center": rows.center.iloc[0]}
        if method in METHODS[1:]:
            row = official_wgg.set_index("method").loc[method]
            for column in ("median_picp90", "min_picp90", "n_picp_lt_0_85",
                           "n_picp_lt_0_80", "median_mpiw"):
                require(abs(summary[column]-row[column]) <= 1e-6,
                        f"official_interval_summary_matches:{method}:{column}")
        summaries.append(summary)
    sm = pd.DataFrame(summaries)
    sm.to_csv(DATA / "figure_5_interval_summary_common_mpiw.csv", index=False)
    return all_rows, sm.set_index("method")


def interval_figure(all_rows, sm):
    fig = plt.figure(figsize=(width_mm / 25.4, 127 / 25.4))
    ax = fig.add_axes([0.14, 0.595, 0.815, 0.255])
    for offset, method, color, marker in zip([-0.18, 0, 0.18], METHODS, COLORS, MARKERS):
        rows = all_rows.loc[all_rows.method == method].sort_values("display_index_by_native_picp")
        ax.scatter(rows.display_index_by_native_picp+offset, rows.picp90,
                   color=color, marker=marker, s=15, alpha=0.9,
                   edgecolors="white", linewidth=0.25, zorder=3)
    for value, ls, color in ((0.90, "--", INK), (0.85, ":", GREY), (0.80, ":", RED)):
        ax.axhline(value, color=color, linestyle=ls, linewidth=0.85)
        ax.text(36.8, value, f"{value:.2f}", ha="left", va="center", fontsize=8, color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.set(xlim=(0, 39), ylim=(0.65, 1.025), ylabel="PICP90",
           xlabel="Background display index (ordered by native PICP90)")
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
    ax.set_yticks([0.7, 0.8, 0.9, 1])
    style(ax)
    title(ax, "a", "Coverage across 36 Development backgrounds")
    ax = fig.add_axes([0.14, 0.17, 0.355, 0.235])
    for method, color, marker in zip(METHODS, COLORS, MARKERS):
        rows = all_rows.loc[all_rows.method == method]
        ax.scatter(rows.mpiw, rows.picp90, color=color, marker=marker,
                   s=14, alpha=0.47, edgecolors="none", zorder=2)
        ax.scatter(sm.loc[method, "median_mpiw"], sm.loc[method, "median_picp90"],
                   color=color, marker=marker, s=46, edgecolors=INK, linewidth=0.6, zorder=4)
    ax.axhline(0.90, color=INK, linestyle="--", linewidth=0.85)
    ax.set(xlim=(0, all_rows.mpiw.max()*1.08), ylim=(0.65, 1.025),
           xlabel="MPIW\n(Mt CO₂ yr⁻¹)", ylabel="PICP90")
    ax.set_yticks([0.7, 0.8, 0.9, 1])
    style(ax)
    title(ax, "b", "Coverage vs width", subtitle="Large symbols: across-group medians")
    ax = fig.add_axes([0.66, 0.17, 0.295, 0.235])
    x = np.arange(3)
    v85 = sm.loc[list(METHODS), "n_picp_lt_0_85"].to_numpy()
    v80 = sm.loc[list(METHODS), "n_picp_lt_0_80"].to_numpy()
    ax.bar(x-0.17, v85, width=0.30, color=COLORS, edgecolor="white", linewidth=0.4,
           label="PICP < 0.85", zorder=3)
    ax.bar(x+0.17, v80, width=0.30, color=COLORS, alpha=0.38,
           edgecolor=INK, linewidth=0.4, hatch="///", label="PICP < 0.80", zorder=3)
    for offset, values in ((-0.17, v85), (0.17, v80)):
        for xx, value in zip(x+offset, values):
            ax.text(xx, value+0.23, str(value), ha="center", fontsize=8)
    ax.set(ylim=(0, 10.9), ylabel="Number of backgrounds")
    ax.set_xticks(x, ["Native", "A4", "WGG"])
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.legend(loc="upper right", fontsize=8, handlelength=1.1, handletextpad=0.3)
    style(ax)
    title(ax, "c", "Failure counts", subtitle="36 backgrounds per method")
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none",
                      markersize=5, label=label)
               for marker, color, label in zip(MARKERS, COLORS, LABELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.951),
               ncol=3, columnspacing=1.0, handletextpad=0.35, fontsize=8)
    fig.text(0.52, 0.976, "Development interval coverage–efficiency trade-offs",
             ha="center", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.024, "WGG: fixed-scale Development assessment, not external confirmation.",
             ha="center", fontsize=8, color=GREY)
    export(fig, "figure_5_interval_coverage_efficiency")


def main():
    setup()
    bg, effects, wgg, official_wgg, before = load_and_check()
    point_figure(bg, effects)
    risk_figure(bg)
    interval_rows, sm = prepare_intervals(bg, wgg, official_wgg)
    interval_figure(interval_rows, sm)
    after = {name: sha256(SOURCE / name) for name in READ_ALLOWLIST}
    require(before == after, "all_source_files_unchanged_after_plotting")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "RESULTS_FIGURE_METRIC_CORRECTION_AND_REDRAW",
        "source_directory": str(SOURCE.relative_to(ROOT)),
        "input_sha256": before,
        "inputs_unchanged": before == after,
        "source_file_allowlist": list(READ_ALLOWLIST),
        "output_artifacts": artifacts,
        "rendering": {"backend": "python_matplotlib", "font": "Times New Roman",
                      "physical_width_mm": width_mm, "template_basis": "LLNCS textwidth=12.2cm",
                      "png_dpi": 300, "tiff_dpi": 600,
                      "svg_text_as_text": True, "pdf_fonttype": 42},
        "scope": {"new_training": False, "new_inference": False,
                  "new_qhat_fitting": False, "new_bootstrap": False,
                  "new_downloads": False, "new_calibration_or_test_data_read": False,
                  "existing_results_modified": False, "main_manuscript_modified": False},
        "statistical_unit": "36 Development background groups; 6750 repeated realizations per group",
        "correction": "Common MPIW replaces non-comparable native-mean/A4-WGG-median normalized widths",
        "evaluation_caveat": "Native: trained held-out-group LOBO; A4/WGG: conditional on fixed A4 scale, not end-to-end external confirmation",
        "figure_5_summary": sm.reset_index().to_dict(orient="records"),
        "checks": checks,
        "drawing_script_sha256": sha256(Path(__file__)),
    }
    (OUT / "figure_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"figures": 3, "backgrounds": N_GROUPS,
                      "interval_points": len(interval_rows), "checks_passed": len(checks),
                      "source_hashes_unchanged": before == after,
                      "outputs": str(FIG)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
