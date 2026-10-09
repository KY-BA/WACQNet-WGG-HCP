# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(r"D:\CO2\_third")
SRC = ROOT / "outputs" / "wacqnet_wgg_finalization_v1"
OUT = ROOT / "outputs" / "wacqnet_wgg_main_figures_v1"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "source_data"

COLORS = {
    "hpr": "#6B7280",
    "wacq": "#145DA0",
    "no_wind": "#9DB7D5",
    "no_contrast": "#6F93C1",
    "no_gdro": "#466F9E",
    "a4": "#7A6FAC",
    "wgg": "#D9822B",
    "good": "#2E8B57",
    "bad": "#B54845",
    "grid": "#D8DDE5",
    "ink": "#222831",
}

METHOD_LABELS = {
    "WACQ_FULL": "WACQNet",
    "WACQ_NO_WIND": "No wind/template",
    "WACQ_NO_CONTRAST": "No contrast",
    "WACQ_NO_GDRO": "No Group-DRO",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.2,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.4,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel_label(ax, text: str) -> None:
    ax.text(-0.10, 1.04, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def style_ax(ax) -> None:
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, alpha=0.65, zorder=0)
    ax.tick_params(width=0.7, length=3)


def save_all(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, fc, ec, fontsize=7.2, weight="normal"):
    # Plain rectangles avoid platform-specific FancyBboxPatch/Bezier DLL issues
    # while preserving editable vector geometry.
    box = Rectangle(xy, width, height, facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)


def arrow(ax, start, end, color="#6B7280"):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=1.0, solid_capstyle="round")
    ax.plot(end[0], end[1], marker=">", ms=4.5, color=color, markeredgewidth=0)


def figure1_framework() -> None:
    fig = plt.figure(figsize=(7.2, 4.45))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.05, 0.85], hspace=0.34)

    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "a")
    rounded_box(ax, (0.02, 0.28), 0.15, 0.44, "OCO-3 XCO$_2$\nmask + wind", "#EEF3F8", COLORS["hpr"], weight="bold")
    rounded_box(ax, (0.22, 0.28), 0.15, 0.44, "Frozen HPR\nanchor  $\\mu_0$", "#F1F2F4", COLORS["hpr"], weight="bold")
    rounded_box(ax, (0.43, 0.18), 0.25, 0.64, "BTI-WACQNet\nwind-aligned encoding\nfrozen plume template\nforeground−background contrast\nGroup-DRO", "#E7F1FA", COLORS["wacq"], weight="bold")
    rounded_box(ax, (0.76, 0.56), 0.20, 0.27, "Point estimate\n$\\hat{Q}_W$", "#DCECF8", COLORS["wacq"], weight="bold")
    rounded_box(ax, (0.76, 0.17), 0.20, 0.27, "Relative risk score\n$r_W$ (not probability)", "#DCECF8", COLORS["wacq"], weight="bold")
    arrow(ax, (0.17, 0.5), (0.22, 0.5)); arrow(ax, (0.37, 0.5), (0.43, 0.5))
    arrow(ax, (0.68, 0.58), (0.76, 0.68), COLORS["wacq"]); arrow(ax, (0.68, 0.42), (0.76, 0.30), COLORS["wacq"])
    ax.text(0.02, 0.90, "Point-estimation and risk-ranking track", color=COLORS["wacq"], fontsize=8.2, fontweight="bold")

    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "b")
    rounded_box(ax, (0.05, 0.28), 0.17, 0.44, "Frozen HPR\ncenter  $\\mu_0$", "#F1F2F4", COLORS["hpr"], weight="bold")
    rounded_box(ax, (0.31, 0.23), 0.20, 0.54, "Frozen A4 scale\n$s_{A4}(x)$", "#EEEAF8", COLORS["a4"], weight="bold")
    rounded_box(ax, (0.60, 0.18), 0.21, 0.64, "WorstGroupGuard-HCP\nmax training-group\n90% tail quantile", "#FCEEDC", COLORS["wgg"], weight="bold")
    rounded_box(ax, (0.86, 0.28), 0.11, 0.44, "Conservative\ninterval", "#F9E1C3", COLORS["wgg"], weight="bold")
    arrow(ax, (0.22, 0.5), (0.31, 0.5)); arrow(ax, (0.51, 0.5), (0.60, 0.5)); arrow(ax, (0.81, 0.5), (0.86, 0.5))
    ax.text(0.05, 0.90, "Conservative interval track (HPR-centered)", color=COLORS["wgg"], fontsize=8.2, fontweight="bold")
    ax.text(0.50, 0.04, "The interval is not centered on WACQNet; the two tracks answer different trust questions.", ha="center", color=COLORS["bad"], fontsize=7.2, fontweight="bold")

    ax = fig.add_subplot(gs[2])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "c")
    xs = [0.03, 0.27, 0.51, 0.75]
    labels = [
        "36 background groups\nDevelopment only",
        "35 groups\nmodel fitting",
        "disjoint repetitions\nfold-internal calibration",
        "1 complete held-out group\n6,750 realizations",
    ]
    colors = ["#EEF3F8", "#E7F1FA", "#EEEAF8", "#FCEEDC"]
    edges = [COLORS["hpr"], COLORS["wacq"], COLORS["a4"], COLORS["wgg"]]
    for i, (x, label, fc, ec) in enumerate(zip(xs, labels, colors, edges)):
        rounded_box(ax, (x, 0.25), 0.20, 0.50, label, fc, ec, fontsize=6.9, weight="bold" if i in (0, 3) else "normal")
        if i < 3:
            arrow(ax, (x + 0.20, 0.50), (xs[i + 1], 0.50))
    ax.text(0.50, 0.06, "Statistical unit: background group; realizations are within-background controlled repeats.", ha="center", fontsize=7.1, fontweight="bold")
    save_all(fig, "figure_1_dual_track_framework")


def figure2_point_effects(bg: pd.DataFrame, effects: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    ax = axes[0, 0]
    ax.scatter(bg["hpr_median_ape"], bg["wacq_median_ape"], s=20, color=COLORS["wacq"], alpha=0.82, edgecolor="white", linewidth=0.35)
    lim = (0, max(bg["hpr_median_ape"].max(), bg["wacq_median_ape"].max()) * 1.04)
    ax.plot(lim, lim, ls="--", color=COLORS["hpr"], lw=0.9)
    ax.set(xlim=lim, ylim=lim, xlabel="Frozen HPR median APE", ylabel="WACQNet median APE", title="Point-error transfer by held-out background")
    ax.text(0.04, 0.93, "36/36 below identity", transform=ax.transAxes, color=COLORS["good"], fontweight="bold")
    panel_label(ax, "a"); style_ax(ax)

    ax = axes[0, 1]
    ax.scatter(bg["hpr_p_reliable30"], bg["wacq_p_reliable30"], s=20, color=COLORS["wacq"], alpha=0.82, edgecolor="white", linewidth=0.35)
    ax.plot([0, 1], [0, 1], ls="--", color=COLORS["hpr"], lw=0.9)
    ax.set(xlim=(0, 0.9), ylim=(0, 0.9), xlabel="Frozen HPR  P(reliable$_{30}$)", ylabel="WACQNet  P(reliable$_{30}$)", title="Operational reliability")
    panel_label(ax, "b"); style_ax(ax)

    ax = axes[1, 0]
    ax.scatter(bg["hpr_scale_slope"], bg["wacq_scale_slope"], s=20, color=COLORS["wacq"], alpha=0.82, edgecolor="white", linewidth=0.35)
    lo = min(bg["hpr_scale_slope"].min(), bg["wacq_scale_slope"].min(), 0)
    hi = max(bg["hpr_scale_slope"].max(), bg["wacq_scale_slope"].max(), 1) * 1.04
    ax.plot([lo, hi], [lo, hi], ls="--", color=COLORS["hpr"], lw=0.9)
    ax.axhline(1.0, color=COLORS["good"], ls=":", lw=0.9)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Frozen HPR slope", ylabel="WACQNet slope", title="Scale-compression slope")
    panel_label(ax, "c"); style_ax(ax)

    ax = axes[1, 1]
    labels = ["Median APE", "P(reliable$_{30}$)", "Scale slope"]
    effect_order = [
        "WACQ-HPR background median APE",
        "WACQ-HPR P_reliable30",
        "WACQ-HPR scale-compression slope",
    ]
    ee = effects.set_index("effect").loc[effect_order]
    y = np.arange(3)[::-1]
    colors = [COLORS["good"]] * 3
    for yi, (_, row), col in zip(y, ee.iterrows(), colors):
        ax.plot([row["bootstrap_95ci_low"], row["bootstrap_95ci_high"]], [yi, yi], color=col, lw=2.0)
        ax.plot(row["median_effect"], yi, "o", color=col, ms=5)
    ax.axvline(0, color=COLORS["hpr"], ls="--", lw=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired background-level effect (WACQNet − HPR)")
    ax.set_title("Median effect with background bootstrap 95% CI")
    panel_label(ax, "d"); style_ax(ax)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.10, wspace=0.35, hspace=0.42)
    save_all(fig, "figure_2_point_estimation_and_effects")


def boxplot_colored(ax, arrays, labels, colors, ylabel, title, ref=None):
    bp = ax.boxplot(arrays, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.62,
                    medianprops=dict(color="#111111", lw=1.1), whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8), boxprops=dict(lw=0.8))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.78)
    # Display every background-level observation. The deterministic modular
    # horizontal spread is only for visibility and does not alter the data.
    for pos, (values, color) in enumerate(zip(arrays, colors), start=1):
        values = np.asarray(values, dtype=float)
        n = len(values)
        order = (np.arange(n) * 17 + pos * 7) % n
        jitter = (order - (n - 1) / 2) * (0.22 / max(n - 1, 1))
        ax.scatter(pos + jitter, values, s=5.5, color=color, alpha=0.45,
                   edgecolor="white", linewidth=0.2, zorder=3)
    if ref is not None:
        ax.axhline(ref, color=COLORS["hpr"], ls="--", lw=0.9)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.tick_params(axis="x", rotation=18)
    style_ax(ax)


def figure3_ablations(bg: pd.DataFrame, metrics: pd.DataFrame, abl_bg: pd.DataFrame) -> None:
    order = ["WACQ_FULL", "WACQ_NO_WIND", "WACQ_NO_CONTRAST", "WACQ_NO_GDRO"]
    labels = [METHOD_LABELS[x] for x in order]
    colors = [COLORS["wacq"], COLORS["no_wind"], COLORS["no_contrast"], COLORS["no_gdro"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15))

    ape_arrays = [bg["hpr_median_ape"].to_numpy()] + [abl_bg.loc[abl_bg.candidate_id == c, "median_ape"].to_numpy() for c in order]
    boxplot_colored(axes[0, 0], ape_arrays, ["HPR"] + labels, [COLORS["hpr"]] + colors, "Background median APE", "Point-estimation ablation")
    panel_label(axes[0, 0], "a")

    risk_arrays = [abl_bg.loc[abl_bg.candidate_id == c, "spearman_risk_ape"].to_numpy() for c in order]
    boxplot_colored(axes[0, 1], risk_arrays, labels, colors, "Spearman(risk, APE)", "Risk-ranking ablation", ref=0)
    panel_label(axes[0, 1], "b")

    width_arrays = [abl_bg.loc[abl_bg.candidate_id == c, "normalized_width"].to_numpy() for c in order]
    boxplot_colored(axes[1, 0], width_arrays, labels, colors, "Normalized interval width", "Native interval efficiency")
    panel_label(axes[1, 0], "c")

    ax = axes[1, 1]
    mm = metrics.set_index("candidate_id").loc[order]
    x = np.arange(len(order)); width = 0.34
    ax.bar(x - width/2, mm["median_picp90"], width, color=colors, alpha=0.78, edgecolor="white", label="Median PICP")
    ax.bar(x + width/2, mm["min_picp90"], width, color=colors, alpha=0.38, edgecolor=COLORS["ink"], hatch="//", label="Worst-group PICP")
    ax.axhline(0.90, color=COLORS["ink"], ls="--", lw=0.9)
    ax.axhline(0.80, color=COLORS["bad"], ls=":", lw=1.0)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("PICP at 90% nominal")
    ax.set_xticks(x, labels, rotation=18); ax.set_title("Macro metrics can hide group failures")
    ax.legend(loc="lower left", fontsize=6.1)
    panel_label(ax, "d"); style_ax(ax)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.13, wspace=0.33, hspace=0.43)
    save_all(fig, "figure_3_preregistered_ablations")


def figure4_interval_tradeoff(wacq_full: pd.DataFrame, wgg_bg: pd.DataFrame, wgg_summary: pd.DataFrame) -> None:
    native = wacq_full[["heldout_background", "picp90", "mpiw", "normalized_width"]].rename(columns={
        "heldout_background": "background_id", "normalized_width": "median_normalized_width"
    })
    native["method"] = "WACQ_NATIVE"
    all_bg = pd.concat([native, wgg_bg], ignore_index=True)
    order_ids = native.sort_values("picp90")["background_id"].tolist()
    rank = {bgid: i + 1 for i, bgid in enumerate(order_ids)}
    all_bg["rank_by_wacq_native"] = all_bg["background_id"].map(rank)
    all_bg.to_csv(DATA / "figure_4_interval_tradeoff.csv", index=False)

    summary_native = pd.DataFrame([{
        "method": "WACQ_NATIVE", "backgrounds": 36,
        "median_picp90": native.picp90.median(),
        "n_picp_lt_0_85": int((native.picp90 < 0.85).sum()),
        "n_picp_lt_0_80": int((native.picp90 < 0.80).sum()),
        "min_picp90": native.picp90.min(),
        "median_mpiw": native.mpiw.median(),
        "median_normalized_width": native.median_normalized_width.median(),
    }])
    summary = pd.concat([summary_native, wgg_summary], ignore_index=True)
    morder = ["WACQ_NATIVE", "A4_FROZEN_REFERENCE", "BTI_WORST_GROUP_GUARD_HCP"]
    labels = ["WACQ native", "A4 (HPR-centered)", "WGG-HCP (HPR-centered)"]
    colors = [COLORS["wacq"], COLORS["a4"], COLORS["wgg"]]
    sm = summary.set_index("method").loc[morder]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    ax = axes[0, 0]
    for method, label, color in zip(morder, labels, colors):
        g = all_bg[all_bg.method == method].sort_values("rank_by_wacq_native")
        ax.plot(g.rank_by_wacq_native, g.picp90, marker="o", ms=2.6, lw=1.0, color=color, alpha=0.88, label=label)
    ax.axhline(0.90, color=COLORS["ink"], ls="--", lw=0.9)
    ax.axhline(0.85, color=COLORS["hpr"], ls=":", lw=0.8)
    ax.axhline(0.80, color=COLORS["bad"], ls=":", lw=0.9)
    ax.set(xlabel="Held-out background rank (sorted by WACQ native PICP)", ylabel="PICP", ylim=(0.55, 1.02), title="Coverage across 36 held-out backgrounds")
    ax.legend(loc="lower right", fontsize=5.6)
    panel_label(ax, "a"); style_ax(ax)

    ax = axes[0, 1]
    for method, label, color in zip(morder, labels, colors):
        g = all_bg[all_bg.method == method]
        ax.scatter(g.median_normalized_width, g.picp90, s=12, color=color, alpha=0.35, edgecolor="none")
        ax.scatter(sm.loc[method, "median_normalized_width"], sm.loc[method, "median_picp90"], s=55, color=color, edgecolor="white", linewidth=0.7, label=label, zorder=4)
    ax.axhline(0.90, color=COLORS["ink"], ls="--", lw=0.9)
    ax.set(xlabel="Normalized interval width", ylabel="PICP", ylim=(0.55, 1.02), title="Coverage–efficiency trade-off")
    ax.legend(loc="lower right", fontsize=5.6)
    panel_label(ax, "b"); style_ax(ax)

    ax = axes[1, 0]
    arrays = [all_bg.loc[all_bg.method == m, "picp90"].to_numpy() for m in morder]
    boxplot_colored(ax, arrays, labels, colors, "PICP", "Background-level coverage distributions", ref=0.90)
    ax.axhline(0.80, color=COLORS["bad"], ls=":", lw=0.9)
    ax.set_ylim(0.55, 1.02)
    panel_label(ax, "c")

    ax = axes[1, 1]
    x = np.arange(3); width = 0.34
    ax.bar(x - width/2, sm["n_picp_lt_0_85"], width, color=colors, alpha=0.78, label="PICP < 0.85")
    ax.bar(x + width/2, sm["n_picp_lt_0_80"], width, color=colors, alpha=0.36, edgecolor=COLORS["ink"], hatch="//", label="PICP < 0.80")
    for i, method in enumerate(morder):
        ax.text(i, max(sm.loc[method, "n_picp_lt_0_85"], sm.loc[method, "n_picp_lt_0_80"]) + 0.25,
                f"MPIW {sm.loc[method, 'median_mpiw']:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x, labels, rotation=18); ax.set_ylabel("Number of backgrounds")
    ax.set_ylim(0, max(sm.n_picp_lt_0_85.max(), 8) + 2.2); ax.set_title("Undercoverage failures and width cost")
    ax.legend(loc="upper right", fontsize=6.0)
    panel_label(ax, "d"); style_ax(ax)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.14, wspace=0.34, hspace=0.44)
    save_all(fig, "figure_4_coverage_efficiency_tradeoff")
    return summary


def make_tables(bg, metrics, wgg_summary, interval_summary):
    TAB.mkdir(parents=True, exist_ok=True)
    main1 = pd.DataFrame([
        ["Frozen HPR", bg.hpr_median_ape.median(), bg.hpr_p_reliable30.median(), bg.hpr_scale_slope.median(), np.nan, np.nan],
        ["WACQNet", metrics.set_index("candidate_id").loc["WACQ_FULL", "median_ape"], metrics.set_index("candidate_id").loc["WACQ_FULL", "median_p_reliable30"], metrics.set_index("candidate_id").loc["WACQ_FULL", "median_scale_slope"], metrics.set_index("candidate_id").loc["WACQ_FULL", "median_risk_spearman"], metrics.set_index("candidate_id").loc["WACQ_FULL", "median_auprc"]],
        ["No wind/template", *metrics.set_index("candidate_id").loc["WACQ_NO_WIND", ["median_ape", "median_p_reliable30", "median_scale_slope", "median_risk_spearman", "median_auprc"]].tolist()],
        ["No contrast", *metrics.set_index("candidate_id").loc["WACQ_NO_CONTRAST", ["median_ape", "median_p_reliable30", "median_scale_slope", "median_risk_spearman", "median_auprc"]].tolist()],
        ["No Group-DRO", *metrics.set_index("candidate_id").loc["WACQ_NO_GDRO", ["median_ape", "median_p_reliable30", "median_scale_slope", "median_risk_spearman", "median_auprc"]].tolist()],
    ], columns=["Method", "Median APE ↓", "Median P(reliable30) ↑", "Median scale slope →1", "Median Spearman(risk, APE) ↑", "Median AUPRC ↑"])
    main1.to_csv(TAB / "table_1_point_risk_ablation.csv", index=False)

    main2 = interval_summary.copy()
    map_names = {"WACQ_NATIVE": "WACQNet native (WACQ-centered)", "A4_FROZEN_REFERENCE": "A4 (HPR-centered)", "BTI_WORST_GROUP_GUARD_HCP": "WorstGroupGuard-HCP (HPR-centered)"}
    main2["Method"] = main2["method"].map(map_names)
    main2 = main2[["Method", "median_picp90", "n_picp_lt_0_85", "n_picp_lt_0_80", "min_picp90", "median_mpiw", "median_normalized_width"]]
    main2.columns = ["Method", "Median PICP90 ↑", "N(PICP<0.85) ↓", "N(PICP<0.80) ↓", "Worst PICP90 ↑", "Median MPIW ↓", "Median normalized width ↓"]
    main2.to_csv(TAB / "table_2_interval_coverage_efficiency.csv", index=False)

    def fmt(v):
        if pd.isna(v): return "—"
        if isinstance(v, (int, np.integer)): return str(int(v))
        return f"{float(v):.4f}"

    def to_md(df):
        header = "| " + " | ".join(df.columns) + " |\n"
        sep = "| " + " | ".join(["---"] + ["---:"] * (len(df.columns) - 1)) + " |\n"
        body = "".join("| " + " | ".join(fmt(v) if j else str(v) for j, v in enumerate(row)) + " |\n" for row in df.itertuples(index=False, name=None))
        return header + sep + body

    md = "# Main tables\n\n## Table 1 | Point estimation, relative-risk ranking and preregistered ablations\n\n" + to_md(main1) + "\nIndependent unit: 36 held-out background groups. Values are medians across backgrounds.\n\n## Table 2 | Interval coverage and efficiency\n\n" + to_md(main2) + "\nWACQNet native intervals are WACQ-centered; A4 and WGG-HCP intervals are HPR-centered and must not be interpreted as the same interval system.\n"
    (TAB / "main_tables.md").write_text(md, encoding="utf-8")

    def latex_escape(s):
        return str(s).replace("_", r"\_").replace("<", r"$<$").replace(">", r"$>$")

    def fmt_tex(v):
        value = fmt(v)
        return r"\textemdash{}" if value == "—" else value

    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Point-estimation and relative-risk results across 36 held-out background groups.}", r"\label{tab:point-risk}", r"\begin{tabular}{lrrrrr}", r"\toprule", "Method & Median APE $\\downarrow$ & $P(\\mathrm{reliable}_{30})$ $\\uparrow$ & Slope $\\to 1$ & Spearman $\\uparrow$ & AUPRC $\\uparrow$ " + r"\\", r"\midrule"]
    for row in main1.itertuples(index=False, name=None):
        lines.append(" & ".join([latex_escape(row[0])] + [fmt_tex(v) for v in row[1:]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", "", r"\begin{table*}[t]", r"\centering", r"\caption{Background-level interval coverage and efficiency. WACQNet and HPR-centered methods use different interval centers.}", r"\label{tab:interval}", r"\begin{tabular}{lrrrrrr}", r"\toprule", "Method & Median PICP & $N(<0.85)$ & $N(<0.80)$ & Worst PICP & Median MPIW & Norm. width " + r"\\", r"\midrule"]
    for row in main2.itertuples(index=False, name=None):
        lines.append(" & ".join([latex_escape(row[0])] + [fmt_tex(v) for v in row[1:]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TAB / "main_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return main1, main2


def write_documentation():
    contract = """# Main figure contract

## Figure 1
- Core conclusion: The final method is an explicitly disclosed dual-track system, not one end-to-end interval model.
- Archetype: schematic-led composite.
- Evidence: deployable inputs, output ownership, 36-background LOBO design.
- Reviewer risk: confusing HPR-centered intervals with WACQNet-centered intervals.

## Figure 2
- Core conclusion: WACQNet improves held-out-background point retrieval and reduces scale compression.
- Archetype: quantitative grid.
- Evidence: 36 paired background summaries and background bootstrap confidence intervals.
- Reviewer risk: treating 243,000 realizations as independent n; the plotted n is 36 backgrounds.

## Figure 3
- Core conclusion: wind/template inputs, foreground-background contrast and Group-DRO have distinguishable roles, while macro coverage can hide worst-background failure.
- Archetype: quantitative grid.
- Evidence: preregistered ablation variants only.
- Reviewer risk: selecting an ablation post hoc as the primary method; explicitly prohibited.

## Figure 4
- Core conclusion: WorstGroupGuard-HCP reduces catastrophic background undercoverage but requires wider HPR-centered intervals.
- Archetype: quantitative grid.
- Evidence: 36-background LOBO PICP and width distributions.
- Reviewer risk: coverage-only reporting; all coverage panels are paired with efficiency metrics.

## Export contract
- Double-column width: 183 mm (7.2 in).
- Backend: Python/matplotlib only.
- Editable SVG and PDF plus 300-dpi PNG and 600-dpi TIFF.
- Statistical unit: background group.
"""
    (OUT / "figure_contract.md").write_text(contract, encoding="utf-8")

    legends = """# Main figure legends

**Fig. 1 | Frozen dual-track framework and development evaluation protocol.** a BTI-WACQNet uses deployable OCO-3, wind, mask, frozen plume-template and HPR-anchor inputs to produce a point estimate and a relative-risk score. b The independent conservative interval track retains the frozen HPR center and applies the A4 scale with WorstGroupGuard-HCP; it is not calibrated around the WACQNet point estimate. c All development results use 36-fold leave-one-background-out evaluation. Each held-out background contains 6,750 controlled Level-B realizations, which are within-background repeats rather than independent backgrounds.

**Fig. 2 | Background-level point-estimation improvement.** a–c Paired results for WACQNet and the frozen HPR across 36 held-out background groups. Dashed diagonal lines indicate equality; the dotted horizontal line in c marks unit slope. d Median paired effects and 95% background-bootstrap confidence intervals (20,000 resamples; n=36 independent background groups). Negative change is favorable for APE; positive change is favorable for reliability and slope.

**Fig. 3 | Preregistered WACQNet ablations.** a Background-level median APE for the frozen HPR, the full WACQNet and three preregistered ablations. b Risk–APE Spearman correlation. c Normalized native interval width. d Median and worst-background PICP at 90% nominal coverage. Boxes show the median and interquartile range across 36 held-out background groups; whiskers extend to 1.5×IQR. Boxplot fliers are suppressed, while every background-level observation is overlaid as a point. Ablations are explanatory and were not eligible to replace the preregistered primary model.

**Fig. 4 | Coverage–efficiency trade-off across interval procedures.** a Background-specific PICP for the WACQNet native interval, the HPR-centered A4 reference and HPR-centered WorstGroupGuard-HCP across the same 36 held-out groups. b Background-level PICP versus normalized width; large points show method medians. c Coverage distributions. d Numbers of moderate and catastrophic undercoverage backgrounds, annotated with median interval width (MPIW). WGG-HCP reduces catastrophic undercoverage but increases interval width. WACQNet and HPR-centered intervals have different centers and are not interchangeable.

Source data are provided in the `source_data` directory.
"""
    (OUT / "main_figure_legends.md").write_text(legends, encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    setup_style()
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    bg = pd.read_csv(SRC / "independent_background_recalculation.csv")
    effects = pd.read_csv(SRC / "background_level_effects.csv")
    metrics = pd.read_csv(SRC / "ablation_summary_verified.csv")
    wgg_bg = pd.read_csv(SRC / "wgg_background_recalculation.csv")
    wgg_summary = pd.read_csv(SRC / "wgg_summary_recalculated.csv")
    original_full = pd.read_csv(ROOT / "outputs" / "wacqnet_development_v1" / "wacqnet_lobo_background_metrics.csv")
    original_abl = pd.read_csv(ROOT / "outputs" / "wacqnet_development_v1" / "wacqnet_lobo_background_metrics_ablations.csv")
    abl_bg = pd.concat([original_full, original_abl], ignore_index=True)

    bg.to_csv(DATA / "figure_2_background_point_effects.csv", index=False)
    effects.to_csv(DATA / "figure_2_bootstrap_effects.csv", index=False)
    abl_bg.to_csv(DATA / "figure_3_ablation_background_metrics.csv", index=False)
    metrics.to_csv(DATA / "figure_3_ablation_summary.csv", index=False)
    figure1_framework()
    figure2_point_effects(bg, effects)
    figure3_ablations(bg, metrics, abl_bg)
    interval_summary = figure4_interval_tradeoff(original_full, wgg_bg, wgg_summary)
    interval_summary.to_csv(DATA / "figure_4_interval_summary.csv", index=False)
    make_tables(bg, metrics, wgg_summary, interval_summary)
    write_documentation()

    hashes = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "output_hashes.json":
            hashes[str(path.relative_to(OUT))] = sha256(path)
    (OUT / "output_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    status = {
        "status": "MAIN_FIGURES_AND_TABLES_FINALIZED",
        "main_figures": 4,
        "main_tables": 2,
        "backend": "Python/matplotlib",
        "independent_unit": "background group",
        "independent_backgrounds": 36,
        "training_run": False,
        "model_inference_run": False,
        "new_calibration_rows_read": 0,
        "new_confirmatory_rows_read": 0,
    }
    (OUT / "build_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
