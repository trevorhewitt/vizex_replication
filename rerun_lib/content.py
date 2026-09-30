"""Content categories from the stored GPT-4o labels (thesis 5.2.2.5, 5.3.2.1,
Figure 5.3A). No API call is made; see 015_tables/vlm_content_labels_SOURCE.md.

The percentages and the chi-square test of independence (7 SLS conditions x 4
categories) follow ai_qual.ipynb (vizex_qual, branch laptop), where the
figure was drawn; the chi-square itself is not in that notebook and is
computed here with scipy.stats.chi2_contingency (no continuity correction is
applied for df > 1).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .common import HZ_MAP, VLM_LABELS_CSV

CAT_ORDER = ["Null", "Geometric Only", "Semantic Only", "Mixed"]
COLORS = {"Null": "#9E9E9E", "Geometric Only": "#2D62B3", "Semantic Only": "#A83232", "Mixed": "#8B39AB"}


def load_labels() -> pd.DataFrame:
    lab = pd.read_csv(VLM_LABELS_CSV, dtype={"condition_code": str})
    lab["condition_hz"] = lab["condition_code"].astype(int).map(HZ_MAP).astype(float)
    return lab


def monte_carlo_p(table: np.ndarray, stat: float, n_sim: int = 20000, seed: int = 42) -> float:
    """Permutation p-value for the chi-square statistic with both margins fixed
    (a check on the asymptotic p, since many expected counts are < 5)."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(table.shape[0]), table.sum(axis=1))
    cols = np.repeat(np.arange(table.shape[1]), table.sum(axis=0))
    exp = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    hits = 0
    for _ in range(n_sim):
        perm = rng.permutation(cols)
        t = np.zeros_like(table)
        np.add.at(t, (rows, perm), 1)
        if (((t - exp) ** 2) / exp).sum() >= stat - 1e-9:
            hits += 1
    return (hits + 1) / (n_sim + 1)


def run(sls_trials: pd.DataFrame, n_sim: int = 20000) -> dict:
    lab = load_labels()
    ct = pd.crosstab(lab["condition_hz"], lab["category"]).reindex(columns=CAT_ORDER).fillna(0).astype(int)
    stat, p, dof, expected = chi2_contingency(ct.values, correction=False)
    geo = lab["geometric_present"].astype(bool)
    sem = lab["semantic_present"].astype(bool)
    by = lab.groupby("condition_hz")
    mid = lab["condition_hz"].between(5, 40)
    res = {
        "labels": lab, "crosstab": ct, "chi2": float(stat), "p": float(p), "dof": int(dof),
        "expected_min": float(expected.min()),
        "expected_lt5_frac": float((expected < 5).mean()),
        "n": int(len(lab)),
        "pct_geo_2_5": 100 * geo[lab["condition_hz"] == 2.5].mean(),
        "pct_null_2_5": 100 * (lab.loc[lab["condition_hz"] == 2.5, "category"] == "Null").mean(),
        "pct_geo_5_40": 100 * geo[mid].mean(),
        "pct_null_80": 100 * (lab.loc[lab["condition_hz"] == 80, "category"] == "Null").mean(),
        "pct_null_15": 100 * (lab.loc[lab["condition_hz"] == 15, "category"] == "Null").mean(),
        "pct_semantic_any_sls": 100 * sem.mean(),
        "geo_by_hz": (100 * by["geometric_present"].mean()).to_dict(),
        "null_by_hz": (100 * by["category"].apply(lambda s: (s == "Null").mean())).to_dict(),
        "semantic_by_hz": (100 * by["semantic_present"].mean()).to_dict(),
    }
    res["mc_p"] = monte_carlo_p(ct.values, stat, n_sim=n_sim) if n_sim else np.nan
    # coverage: labels must be exactly the in-analysis SLS trials
    res["labels_minus_sls"] = sorted(set(lab["png_filename"]) - set(sls_trials["png_filename"]))
    res["sls_minus_labels"] = sorted(set(sls_trials["png_filename"]) - set(lab["png_filename"]))
    return res


def plot(ct: pd.DataFrame, out_path) -> None:
    """Figure 5.3A as drawn in ai_qual.ipynb ('better plot' cell)."""
    order = [2.5, 5.0, 10.0, 15.0, 20.0, 40.0, 80.0]
    ct = ct.reindex(index=order)
    prop = ct.div(ct.sum(axis=1), axis=0)
    prop.index = [("2.5" if h == 2.5 else str(int(h))) + "Hz" for h in prop.index]
    prop.columns = ["Null", "Geometric", "Semantic", "Mixed"]
    fig, ax = plt.subplots(figsize=(10, 8))
    prop.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[c] for c in CAT_ORDER],
              edgecolor="none", width=0.85, legend=False)
    ax.set_title("VLM CLASSIFICATIONS", fontsize=24, fontweight="bold", pad=20)
    ax.set_ylabel("PROPORTION", fontsize=20, fontweight="bold")
    ax.set_xlabel("Hz", fontsize=20, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(3)
    plt.xticks(rotation=0, fontsize=18, fontweight="bold")
    plt.yticks(fontsize=18, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False,
              prop={"weight": "bold", "size": 16})
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
