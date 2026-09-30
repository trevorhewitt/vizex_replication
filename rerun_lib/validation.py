"""Validation analysis (thesis 5.2.2.4, 5.3.1, Table S5.3, Figure 5.1).

Port of 011_notebooks_v2/030_feature_validation_analysis.ipynb:
recreation ~ target (ML, random participant intercept), Holm across the five
features; categorical robustness model recreation ~ C(image); five figure panels.
"""

import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests

from .common import FEATURE_LABELS, FEATURES, fit_mixedlm


def is_validation_code(x) -> bool:
    return bool(re.fullmatch(r"[ABC]\d{2}", str(x).strip()))


def validation_sort_key(code):
    m = re.fullmatch(r"([A-Z])(\d+)", str(code).strip())
    return (ord(m.group(1)), int(m.group(2)), str(code)) if m else (999, 999, str(code))


def prepare_validation_df(trials: pd.DataFrame, targets: pd.DataFrame, feature: str) -> pd.DataFrame:
    rec = trials[trials["stimulus_type"].astype(str).str.upper().eq("IMAGE")]
    rec = rec[rec["condition_code"].map(is_validation_code)]
    targ = targets[targets["condition_code"].map(is_validation_code)]
    df = pd.merge(
        rec[["condition_code", "participant_code", feature]].rename(columns={feature: f"{feature}_rec"}),
        targ[["condition_code", feature]].rename(columns={feature: f"{feature}_targ"}),
        on="condition_code", how="left",
    ).dropna(subset=[f"{feature}_rec", f"{feature}_targ"])
    df["cond_order"] = df["condition_code"].map(validation_sort_key)
    return (df.sort_values(["cond_order", "participant_code"])
              .drop(columns=["cond_order"]).reset_index(drop=True))


def run(trials: pd.DataFrame, targets: pd.DataFrame) -> dict:
    vdfs = {f: prepare_validation_df(trials, targets, f) for f in FEATURES}
    rows = []
    for f in FEATURES:
        d = vdfs[f]
        md, r, method = fit_mixedlm(f"{f}_rec ~ {f}_targ", d)
        key = f"{f}_targ"
        dt, dn = float(r.cov_re.iloc[0, 0]), float(r.scale)
        rows.append({
            "Feature": FEATURE_LABELS[f], "feature": f, "n": int(len(d)),
            "n_participants": int(d["participant_code"].nunique()),
            "n_images": int(d["condition_code"].nunique()),
            "b": float(r.params[key]), "se": float(r.bse[key]), "z": float(r.tvalues[key]),
            "p_unc": float(r.pvalues[key]), "DT": dt, "DN": dn, "ICC": dt / (dt + dn),
            "Signal_Val": float(np.var(md.predict(r.params))),
            "optimiser": method, "converged": bool(getattr(r, "converged", False)),
            "llf": float(r.llf),
        })
    raw = pd.DataFrame(rows)
    raw["p_holm"] = multipletests(raw["p_unc"], alpha=0.05, method="holm")[1]

    cat_rows = []
    for f in FEATURES:
        d = vdfs[f]
        md, r, method = fit_mixedlm(f"{f}_rec ~ C(condition_code)", d)
        t = r.wald_test_terms(scalar=True).table if _has_scalar_kw(r) else r.wald_test_terms().table
        cat_rows.append({
            "Feature": FEATURE_LABELS[f], "n": int(len(d)),
            "Chi2": float(np.squeeze(t.loc["C(condition_code)", "statistic"])),
            "df": int(np.squeeze(t.loc["C(condition_code)", "df_constraint"])),
            "p_unc": float(np.squeeze(t.loc["C(condition_code)", "pvalue"])),
            "DT": float(r.cov_re.iloc[0, 0]), "DN": float(r.scale), "optimiser": method,
        })
    cat = pd.DataFrame(cat_rows)
    cat["p_holm"] = multipletests(cat["p_unc"].fillna(1.0), alpha=0.05, method="holm")[1]
    return {"raw": raw, "categorical": cat, "dfs": vdfs}


def _has_scalar_kw(res) -> bool:
    import inspect
    try:
        return "scalar" in inspect.signature(res.wald_test_terms).parameters
    except (TypeError, ValueError):
        return False


def paper_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Table S5.3 layout and rounding."""
    from .common import format_p
    return pd.DataFrame({
        "Feature": raw["Feature"],
        "n": raw["n"],
        "b (slope)": raw["b"].map(lambda v: f"{v:.6f}"),
        "SE": raw["se"].map(lambda v: f"{v:.6f}"),
        "z": raw["z"].map(lambda v: f"{v:.2f}"),
        "p (Holm)": raw["p_holm"].map(format_p),
        "Random-intercept variance": raw["DT"].map(lambda v: f"{v:.6f}"),
        "Residual variance": raw["DN"].map(lambda v: f"{v:.6f}"),
    })


# ---------------------------------------------------------------- Figure 5.1
FIG_SIZE = (5.0, 4.4)
FIG_DPI = 300
AXES_RECT = (0.26, 0.24, 0.56, 0.62)
AXIS_LABEL_WRAP = 20
PANEL_LETTERS = {"detail": "A", "r_directionality": "B", "theta_directionality": "C",
                 "brightness": "D", "contrast": "E"}
FONT_SIZE_TITLE, FONT_SIZE_AXIS_LABEL, FONT_SIZE_TICK = 19, 13, 13
LINE_WIDTH_SPINE = LINE_WIDTH_TICK = 2.8
TICK_LENGTH = 6
LINE_WIDTH_FIT, LINE_WIDTH_IDENTITY = 4.0, 3.5
IDENTITY_DASH = (0, (2, 1.6))
MARKER_SIZE, MARKER_ALPHA = 26, 0.28
COLOR_POINTS, COLOR_FIT, COLOR_IDENTITY = "#000000", "#003366", "#E8720C"
N_TICKS, TICK_DECIMALS, AXIS_PAD_FRAC = 3, 2, 0.04


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(LINE_WIDTH_SPINE)
    ax.tick_params(axis="both", width=LINE_WIDTH_TICK, length=TICK_LENGTH, labelsize=FONT_SIZE_TICK)
    ax.grid(False)


def plot_panels(vdfs: dict, out_dir) -> list:
    """Figure 5.1 A-E. As in 030, the drawn line is the OLS fit of recreation on
    stimulus (not the LME slope); see the audit notes."""
    paths = []
    for f in FEATURES:
        df = vdfs[f]
        x = df[f"{f}_targ"].to_numpy(float)
        y = df[f"{f}_rec"].to_numpy(float)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = AXIS_PAD_FRAC * (hi - lo) if hi > lo else 0.05
        lo_lim = lo - pad
        if lo >= 0 and lo_lim < 0:
            lo_lim = 0.0
        lim = (lo_lim, hi + pad)
        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_axes(AXES_RECT)
        ax.set_box_aspect(1)
        ax.scatter(x, y, s=MARKER_SIZE, color=COLOR_POINTS, alpha=MARKER_ALPHA, edgecolors="none", zorder=1)
        ax.plot(lim, lim, color=COLOR_IDENTITY, linewidth=LINE_WIDTH_IDENTITY,
                linestyle=IDENTITY_DASH, solid_capstyle="butt", zorder=2)
        m = smf.ols(f"{f}_rec ~ {f}_targ", data=df).fit()
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(xx, m.params["Intercept"] + m.params[f"{f}_targ"] * xx, color=COLOR_FIT,
                linewidth=LINE_WIDTH_FIT, solid_capstyle="round", zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ticks = np.linspace(lim[0], lim[1], N_TICKS)
        labels = [f"{t:.{TICK_DECIMALS}f}" for t in ticks]
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=0)
        ax.set_yticks(ticks); ax.set_yticklabels(labels)
        ax.set_title(f"{PANEL_LETTERS[f]}. {FEATURE_LABELS[f]}", fontsize=FONT_SIZE_TITLE,
                     fontweight="bold", loc="left", pad=10)
        ax.set_xlabel(textwrap.fill("stimulus value", AXIS_LABEL_WRAP), fontsize=FONT_SIZE_AXIS_LABEL,
                      fontweight="bold", labelpad=7)
        ax.set_ylabel(textwrap.fill("recreation value", AXIS_LABEL_WRAP), fontsize=FONT_SIZE_AXIS_LABEL,
                      fontweight="bold", labelpad=7)
        _style_axes(ax)
        p = out_dir / f"fig5.1_{PANEL_LETTERS[f]}_val_{f}.png"
        fig.savefig(p, dpi=FIG_DPI, transparent=False)
        plt.close(fig)
        paths.append(p)

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=np.sqrt(MARKER_SIZE) * 1.9,
               color=COLOR_POINTS, alpha=min(MARKER_ALPHA * 2.2, 1.0), label="individual trial"),
        Line2D([], [], color=COLOR_FIT, linewidth=LINE_WIDTH_FIT, label="model fit"),
        Line2D([], [], color=COLOR_IDENTITY, linewidth=LINE_WIDTH_IDENTITY, linestyle=IDENTITY_DASH,
               label="ideal response"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 0.9))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=14, handlelength=2.4,
              columnspacing=2.0, handletextpad=0.8, borderpad=0.0)
    p = out_dir / "fig5.1_key.png"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    paths.append(p)
    return paths
