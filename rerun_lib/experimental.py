"""Experimental analysis (thesis 5.2.2.6-7, 5.3.2.2-3, Table S5.4, Figure 5.3B-F).

Port of 011_notebooks_v2/035_feature_experimental_analysis.ipynb:
feature ~ C(condition_hz) over all SLS trials (7 levels incl. the 80 Hz control;
ML, random participant intercept), Wald chi-square on the frequency term, Holm
across the five features; the planned contrasts (Experimental vs Control and the
10 vs 15 Hz follow-up); condition means/SDs; five figure panels.
"""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from statsmodels.stats.multitest import multipletests

from .common import FEATURE_LABELS, FEATURES, fit_mixedlm, format_p
from .validation import _has_scalar_kw


def strobe_frame(trials: pd.DataFrame) -> pd.DataFrame:
    df = trials[trials["stimulus_type"] == "LIGHT"].copy()
    for col in FEATURES + ["condition_hz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["condition_hz"] = df["condition_hz"].round(2)
    df = df.dropna(subset=FEATURES + ["condition_hz", "participant_code"]).copy()
    levels = sorted(df["condition_hz"].unique().tolist())
    df["hz_cat"] = pd.Categorical(df["condition_hz"], categories=levels, ordered=True)
    assert (df["hz_cat"].astype(float) == df["condition_hz"]).all(), "hz_cat misaligned"
    return df


def omnibus(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, coefs = [], []
    for f in FEATURES:
        md, r, method = fit_mixedlm(f"{f} ~ C(condition_hz)", df)
        t = r.wald_test_terms(scalar=True).table if _has_scalar_kw(r) else r.wald_test_terms().table
        chi2 = float(np.squeeze(t.loc["C(condition_hz)", "statistic"]))
        dfc = int(np.squeeze(t.loc["C(condition_hz)", "df_constraint"]))
        p = float(np.squeeze(t.loc["C(condition_hz)", "pvalue"]))
        dt, dn = float(r.cov_re.iloc[0, 0]), float(r.scale)
        mean = float(df[f].mean())
        sd = float(np.sqrt(dn))
        rows.append({
            "Feature": FEATURE_LABELS[f], "feature": f, "n": int(len(df)),
            "n_participants": int(df["participant_code"].nunique()),
            "Mean": mean, "Chi2": chi2, "df": dfc, "p_unc": p, "Trial_SD": sd,
            "Trial_CV%": sd / mean * 100 if mean else np.nan, "ICC": dt / (dt + dn),
            "DT": dt, "DN": dn, "Signal_Exp": float(np.var(md.predict(r.params))),
            "optimiser": method, "converged": bool(getattr(r, "converged", False)),
            "llf": float(r.llf),
        })
        for name in r.fe_params.index:
            coefs.append({"feature": f, "term": name, "b": float(r.fe_params[name]),
                          "se": float(r.bse_fe[name]), "z": float(r.tvalues[name]),
                          "p": float(r.pvalues[name])})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_unc"].fillna(1.0), alpha=0.05, method="holm")[1]
    return out, pd.DataFrame(coefs)


CONTRASTS = [
    {"label": "Experimental vs Control", "reference": "Control",
     "subset": lambda d: d[d["condition_type"].isin(["EXPERIMENTAL", "CONTROL"])],
     "group": lambda d: np.where(d["condition_type"].eq("EXPERIMENTAL"), "Experimental", "Control")},
    {"label": "15 Hz vs 10 Hz", "reference": "10 Hz",
     "subset": lambda d: d[d["condition_hz"].isin([10, 15])],
     "group": lambda d: np.where(d["condition_hz"].eq(15), "15 Hz", "10 Hz")},
]


def contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        for spec in CONTRASTS:
            d = spec["subset"](df).copy()
            d["contrast_group"] = spec["group"](d)
            md, r, method = fit_mixedlm(
                f"{f} ~ C(contrast_group, Treatment(reference='{spec['reference']}'))", d)
            key = [k for k in r.params.index if k.startswith("C(contrast_group")][0]
            dt, dn = float(r.cov_re.iloc[0, 0]), float(r.scale)
            rows.append({"Feature": FEATURE_LABELS[f], "feature": f, "Contrast": spec["label"],
                         "n": int(len(d)), "n_participants": int(d["participant_code"].nunique()),
                         "b": float(r.params[key]), "se": float(r.bse[key]),
                         "z": float(r.tvalues[key]), "p_unc": float(r.pvalues[key]),
                         "DT": dt, "DN": dn, "ICC": dt / (dt + dn), "optimiser": method,
                         "converged": bool(getattr(r, "converged", False))})
    out = pd.DataFrame(rows)
    # As in 035: Holm across all ten rows. The thesis reports the 10 vs 15 Hz
    # detail test as a single, uncorrected test (5.2.2.3), so p_unc is the
    # thesis-relevant value for that row.
    out["p_holm_10tests"] = multipletests(out["p_unc"], alpha=0.05, method="holm")[1]
    return out


def condition_means(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("condition_hz")
    rows = []
    for hz, d in g:
        row = {"condition_hz": hz, "n": len(d), "n_participants": d["participant_code"].nunique()}
        for f in FEATURES:
            row[f"{f}_mean"] = d[f].mean()
            row[f"{f}_sd"] = d[f].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def paper_table(o: pd.DataFrame) -> pd.DataFrame:
    """Table S5.4 layout and rounding."""
    return pd.DataFrame({
        "Feature": o["Feature"], "n": o["n"], "Mean": o["Mean"].map(lambda v: f"{v:.2f}"),
        "Chi2": o["Chi2"].map(lambda v: f"{v:.1f}"), "df": o["df"],
        "p (Holm)": o["p_holm"].map(format_p),
        "Trial SD": o["Trial_SD"].map(lambda v: f"{v:.2f}"),
        "Trial_CV%": o["Trial_CV%"].map(lambda v: f"{v:.2f}%"),
        "ICC": o["ICC"].map(lambda v: f"{v:.3f}"),
        "Random-intercept variance": o["DT"].map(lambda v: f"{v:.6f}"),
        "Residual variance": o["DN"].map(lambda v: f"{v:.6f}"),
    })


# ------------------------------------------------------------ Figure 5.3 B-F
FIG_SIZE, FIG_DPI = (5.0, 3.9), 300
AXES_RECT = (0.20, 0.22, 0.76, 0.66)
PANEL_LETTERS = {"detail": "B", "r_directionality": "C", "theta_directionality": "D",
                 "brightness": "E", "contrast": "F"}
LINE_WIDTH_SPINE = LINE_WIDTH_TICK = 2.8
LINE_WIDTH_MEAN, LINE_WIDTH_SD, SD_CAP_SIZE = 4.0, 2.8, 6
MARKER_SIZE_MEAN, MARKER_EDGE_WIDTH, MARKER_SIZE_DATA, MARKER_ALPHA, JITTER = 9, 2.5, 18, 0.16, 0.13
COLOR_POINTS, COLOR_MEAN = "#000000", "#003366"
Y_MAX_BY_FEATURE = {"detail": 0.75}


def _fmt_hz(hz) -> str:
    hz = float(hz)
    return str(int(hz)) if hz.is_integer() else str(hz)


def plot_panels(df: pd.DataFrame, out_dir) -> list:
    paths = []
    levels = list(df["hz_cat"].cat.categories)
    x_pos = np.arange(len(levels), dtype=float)
    for f in FEATURES:
        rng = np.random.RandomState(42)   # as in 035
        grouped = df.groupby("condition_hz")[f]
        means = grouped.mean().reindex(levels).to_numpy(float)
        sds = np.nan_to_num(grouped.std(ddof=1).reindex(levels).to_numpy(float), nan=0.0)
        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_axes(AXES_RECT)
        for i, hz in enumerate(levels):
            vals = df.loc[df["condition_hz"] == hz, f].to_numpy(float)
            ax.scatter(x_pos[i] + rng.normal(0, JITTER, size=vals.size), vals, s=MARKER_SIZE_DATA,
                       color=COLOR_POINTS, alpha=MARKER_ALPHA, edgecolors="none", zorder=1, clip_on=True)
        ax.errorbar(x_pos, means, yerr=sds, fmt="none", ecolor=COLOR_MEAN, elinewidth=LINE_WIDTH_SD,
                    capsize=SD_CAP_SIZE, capthick=LINE_WIDTH_SD, zorder=2, clip_on=True)
        ax.plot(x_pos, means, color=COLOR_MEAN, linewidth=LINE_WIDTH_MEAN, marker="o",
                markersize=MARKER_SIZE_MEAN, markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
                markeredgecolor=COLOR_MEAN, solid_capstyle="round", zorder=3, clip_on=False)
        bottom, top = ax.get_ylim()
        if Y_MAX_BY_FEATURE.get(f) is not None:
            top = Y_MAX_BY_FEATURE[f]
        ax.set_ylim(bottom, top)
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([_fmt_hz(h) for h in levels], rotation=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(f"{PANEL_LETTERS[f]}. {FEATURE_LABELS[f]}", fontsize=19, fontweight="bold",
                     loc="left", pad=10)
        ax.set_xlabel(textwrap.fill("stroboscopic frequency (Hz)", 28), fontsize=13, fontweight="bold", labelpad=7)
        ax.set_ylabel(textwrap.fill(FEATURE_LABELS[f], 28), fontsize=13, fontweight="bold", labelpad=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(LINE_WIDTH_SPINE)
        ax.tick_params(axis="both", width=LINE_WIDTH_TICK, length=6, labelsize=13)
        p = out_dir / f"fig5.3_{PANEL_LETTERS[f]}_exp_{f}.png"
        fig.savefig(p, dpi=FIG_DPI, transparent=False)
        plt.close(fig)
        paths.append(p)

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=np.sqrt(MARKER_SIZE_DATA) * 1.9,
               color=COLOR_POINTS, alpha=min(MARKER_ALPHA * 3.0, 1.0), label="individual trial"),
        Line2D([], [], color=COLOR_MEAN, linewidth=LINE_WIDTH_MEAN, marker="o", markersize=MARKER_SIZE_MEAN,
               markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor=COLOR_MEAN,
               label="condition mean"),
        Line2D([], [], color=COLOR_MEAN, linewidth=LINE_WIDTH_SD, marker="_", markersize=10,
               markeredgewidth=LINE_WIDTH_SD, label="± 1 SD"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 0.9))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=14, handlelength=2.4,
              columnspacing=2.0, handletextpad=0.8, borderpad=0.0)
    p = out_dir / "fig5.3_key.png"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    paths.append(p)
    return paths


def run(trials: pd.DataFrame) -> dict:
    df = strobe_frame(trials)
    omni, coefs = omnibus(df)
    con = contrasts(df)
    means = condition_means(df)
    return {"df": df, "omnibus": omni, "coefs": coefs, "contrasts": con, "means": means}
