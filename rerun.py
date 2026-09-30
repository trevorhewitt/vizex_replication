#!/usr/bin/env python
"""Regenerate every Chapter 5 table and figure from the committed CSVs.

    python rerun.py                  # the full rerun (about 3-5 minutes)
    python rerun.py --no-baseline    # skip the pre-exclusion reproduction column
    python rerun.py --with-images    # ALSO run 010/015 on 005_cleaned_data/ (Figure S5.4)
    python rerun.py --notebooks      # ALSO execute 030/035/036 and compare with the scripts

Outputs go to rerun_outputs/ (see RERUN.md). Exit code 0 = every hard check
passed; 1 = a hard check failed (the run stops at the failing stage).
"""

import argparse
import io
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerun_lib import (build_tables, content, experimental, report, thesis_values,  # noqa: E402
                       trial_order, validation, verify)
from rerun_lib.common import (EXPECTED, FEATURE_LABELS, FEATURES, FIG_PATH, OUT_PATH,  # noqa: E402
                              PROJECT_PATH, TAB_PATH, TABLES_PATH, Checks, ensure_dirs,
                              fit_mixedlm, quiet_warnings, set_seeds)

# Commit whose 015_tables hold the pre-exclusion (1,581-trial) data the thesis
# numbers were computed from (identical at 479ccfb and 6e52ac6).
BASELINE_COMMIT = "479ccfb1b44ff6916e6ec0921d430955ac10346c"
LABEL_TO_FEATURE = {v: k for k, v in FEATURE_LABELS.items()}


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


# ----------------------------------------------------------------- analysis
def analyse(data: dict, n_sim: int) -> dict:
    f, tg = data["features"], data["targets"]
    val = validation.run(f, tg)
    exp = experimental.run(f)
    order = trial_order.run(f, tg)
    sls = f[f["stimulus_type"] == "LIGHT"]
    cont = content.run(sls, n_sim=n_sim)
    o = order["order"].copy()
    o["feature"] = o["Feature"].map(LABEL_TO_FEATURE)
    counts = {
        "in_analysis": int(len(f)),
        "experimental": int((f["condition_type"] == "EXPERIMENTAL").sum()),
        "control": int((f["condition_type"] == "CONTROL").sum()),
        "validation": int((f["condition_type"] == "VALIDATION").sum()),
        "participants": int(f["participant_code"].nunique()),
        "sls": int(len(sls)),
        "per_hz": {float(k): int(v) for k, v in sls.groupby("condition_hz").size().items()},
    }
    return {
        "counts": counts, "val": val["raw"].set_index("feature"), "val_full": val,
        "omni": exp["omnibus"].set_index("feature"), "exp_full": exp,
        "con": exp["contrasts"], "means": exp["means"], "order": o, "order_full": order,
        "order_meta": {"n_exp": len(order["exp"]), "n_val": len(order["val"]),
                       "max_shift": order["max_shift"], "min_r": order["min_r"]},
        "content": cont, "scales": data["scales"],
    }


def hard_counts(checks: Checks, data: dict, label: str):
    tm, f = data["trials_master"], data["features"]
    E = EXPECTED
    checks.hard(f"[{label}] trials_master.csv rows", len(tm), E["trials_master_rows"])
    checks.hard(f"[{label}] trials in analysis (in_analysis == True)", int(tm["in_analysis"].sum()), E["in_analysis"])
    excl = tm[~tm["in_analysis"]]
    checks.hard(f"[{label}] excluded trials", len(excl), E["excluded"])
    checks.hard(f"[{label}] excluded trial is P358 trial 8 (manual)",
                sorted(zip(excl["participant_code"], excl["trialn"].astype(int), excl["exclusion_reason"])),
                [("P358", 8, "manual")])
    checks.hard(f"[{label}] vizex_features.csv rows == in-analysis trials", len(f), E["in_analysis"])
    checks.hard(f"[{label}] vizex_features.csv trials == trials_master in-analysis trials",
                set(f["png_filename"]) == set(tm.loc[tm["in_analysis"], "png_filename"]), True)
    ct = f["condition_type"].value_counts().to_dict()
    checks.hard(f"[{label}] experimental trials", ct.get("EXPERIMENTAL", 0), E["experimental"])
    checks.hard(f"[{label}] control (80 Hz) trials", ct.get("CONTROL", 0), E["control"])
    checks.hard(f"[{label}] validation trials", ct.get("VALIDATION", 0), E["validation"])
    checks.hard(f"[{label}] participants", int(f["participant_code"].nunique()), E["participants"])
    sls = f[f["stimulus_type"] == "LIGHT"]
    checks.hard(f"[{label}] SLS trials (experimental + control)", len(sls), E["sls"])
    per_hz = {float(k): int(v) for k, v in sls.groupby("condition_hz").size().items()}
    checks.hard(f"[{label}] trials per frequency (2.5/5/10/15/20/40/80 Hz)", per_hz, E["per_hz"])
    p358 = f[f["participant_code"] == "P358"]
    checks.hard(f"[{label}] P358 trial 9 (blank 40 Hz) kept", bool(((p358["trialn"] == 9) & (p358["condition_hz"] == 40)).any()), True)
    checks.hard(f"[{label}] December 2025 trials (trialN 101/102) kept in the main set",
                sorted(zip(f.loc[f["trialn"] > 17, "participant_code"], f.loc[f["trialn"] > 17, "trialn"].astype(int))),
                [("P306", 101), ("P306", 102), ("P324", 101)])


def model_counts(checks: Checks, R: dict, label: str):
    checks.hard(f"[{label}] validation models: n per feature", sorted(set(R["val"]["n"])), [EXPECTED["validation"]])
    checks.hard(f"[{label}] experimental models (Table S5.4): n per feature", sorted(set(R["omni"]["n"])), [EXPECTED["sls"]])
    checks.hard(f"[{label}] experimental models: frequency levels (df)", sorted(set(R["omni"]["df"])), [6])
    oexp = R["order_full"]["exp"]
    checks.hard(f"[{label}] trial-order model: SLS trials", len(oexp), EXPECTED["sls_trial_order"])
    checks.hard(f"[{label}] trial-order model: experimental / control",
                (int((oexp["condition_type"] == "EXPERIMENTAL").sum()), int((oexp["condition_type"] == "CONTROL").sum())),
                (EXPECTED["experimental_trial_order"], EXPECTED["control"]))
    checks.hard(f"[{label}] trial-order model: validation trials", len(R["order_full"]["val"]), EXPECTED["validation_trial_order"])
    checks.hard(f"[{label}] trial-order: December trials excluded", R["order_full"]["n_late"], 3)
    checks.hard(f"[{label}] trial-order: span of positions (drift = b x span)", R["order_full"]["trial_span"], 15)
    checks.hard(f"[{label}] trial-order: P368 make-up position", R["order_full"]["placed"], [("P368", 1, "C00", 4.5)])
    con = R["con"]
    n1015 = int(con[(con["feature"] == "detail") & (con["Contrast"] == "15 Hz vs 10 Hz")]["n"].iloc[0])
    checks.hard(f"[{label}] 10 vs 15 Hz follow-up: trials", n1015, 395)
    c = R["content"]
    checks.hard(f"[{label}] VLM labels cover exactly the in-analysis SLS trials",
                (len(c["labels_minus_sls"]), len(c["sls_minus_labels"]), c["n"]), (0, 0, EXPECTED["sls"]))


# ----------------------------------------------------------------- outputs
def write_outputs(R: dict):
    val, exp, order, cont = R["val_full"], R["exp_full"], R["order_full"], R["content"]
    validation.paper_table(val["raw"]).to_csv(TAB_PATH / "table_S5.3_validation.csv", index=False)
    val["raw"].to_csv(TAB_PATH / "table_S5.3_validation_raw.csv", index=False)
    val["categorical"].to_csv(TAB_PATH / "validation_categorical_robustness_raw.csv", index=False)
    experimental.paper_table(exp["omnibus"]).to_csv(TAB_PATH / "table_S5.4_experimental.csv", index=False)
    exp["omnibus"].to_csv(TAB_PATH / "table_S5.4_experimental_raw.csv", index=False)
    exp["coefs"].to_csv(TAB_PATH / "experimental_fixed_effects_vs_2.5Hz_raw.csv", index=False)
    exp["contrasts"].to_csv(TAB_PATH / "contrasts_10v15Hz_and_exp_vs_control_raw.csv", index=False)
    exp["means"].to_csv(TAB_PATH / "condition_means_sd.csv", index=False)
    trial_order.paper_table(order["order"]).to_csv(TAB_PATH / "table_S5.2_trial_order_session.csv", index=False)
    R["order"].to_csv(TAB_PATH / "table_S5.2_trial_order_session_raw.csv", index=False)
    order["accuracy"].to_csv(TAB_PATH / "recreation_accuracy_by_trial_raw.csv", index=False)
    ct = cont["crosstab"].copy()
    ct.index.name = "condition_hz"
    ct.to_csv(TAB_PATH / "content_categories_counts.csv")
    prop = (100 * ct.div(ct.sum(axis=1), axis=0)).round(2)
    prop.to_csv(TAB_PATH / "content_categories_percent.csv")
    pd.DataFrame([{
        "chi2": cont["chi2"], "df": cont["dof"], "p": cont["p"], "p_monte_carlo": cont["mc_p"],
        "n": cont["n"], "min_expected": cont["expected_min"], "share_expected_lt5": cont["expected_lt5_frac"],
        "pct_geo_2.5Hz": cont["pct_geo_2_5"], "pct_null_2.5Hz": cont["pct_null_2_5"],
        "pct_geo_5to40Hz": cont["pct_geo_5_40"], "pct_null_80Hz": cont["pct_null_80"],
        "pct_null_15Hz": cont["pct_null_15"], "pct_semantic_any_SLS": cont["pct_semantic_any_sls"],
    }]).to_csv(TAB_PATH / "content_chi_square.csv", index=False)
    pd.DataFrame([{"trials": k, "n": v} for k, v in R["counts"].items() if k != "per_hz"]
                 + [{"trials": f"{k:g} Hz", "n": v} for k, v in R["counts"]["per_hz"].items()]
                 ).to_csv(TAB_PATH / "analysis_set_counts.csv", index=False)
    pd.DataFrame([{"feature": k, "scale_1_over_Q95": v} for k, v in R["scales"].items()]
                 ).to_csv(TAB_PATH / "normalisation_scales.csv", index=False)

    validation.plot_panels(val["dfs"], FIG_PATH)
    experimental.plot_panels(exp["df"], FIG_PATH)
    content.plot(cont["crosstab"], FIG_PATH / "fig5.3_A_content_categories.png")


# ----------------------------------------------------------------- part D
def verification(checks: Checks, R: dict, data: dict) -> pd.DataFrame:
    rows = []
    f = data["features"]
    sdf = R["exp_full"]["df"]

    def rec(what, a, b, tol, note=""):
        d = verify.rel_diff(a, b)
        rows.append({"check": what, "statsmodels / pipeline": a, "independent": b,
                     "rel_diff": d, "tolerance": tol, "status": "PASS" if d <= tol else "FAIL", "note": note})
        checks.soft(f"[D] {what}", d <= tol, f"{a:.6g} vs {b:.6g}", f"rel diff <= {tol:g}")

    for feat in FEATURES:
        o = R["omni"].loc[feat]
        fit = verify.ri_lme(f"{feat} ~ C(condition_hz)", sdf, "participant_code")
        w, dfw, _ = verify.wald(fit, "C(condition_hz)")
        rec(f"S5.4 {feat}: Wald chi2(6), independent ML fit", o["Chi2"], w, 1e-4)
        rec(f"S5.4 {feat}: participant variance", o["DT"], fit["tau2"], 1e-3)
        rec(f"S5.4 {feat}: residual variance", o["DN"], fit["sigma2"], 1e-4)
        rec(f"S5.4 {feat}: ICC", o["ICC"], fit["icc"], 1e-3)
        rec(f"S5.4 {feat}: log-likelihood", o["llf"], fit["llf"], 1e-6)
        # by hand, from the statsmodels estimates
        md, r, _ = fit_mixedlm(f"{feat} ~ C(condition_hz)", sdf)
        idx = [n for n in r.fe_params.index if n.startswith("C(condition_hz)")]
        b = r.fe_params[idx].to_numpy()
        V = r.cov_params().loc[idx, idx].to_numpy()
        rec(f"S5.4 {feat}: Wald chi2 by hand b'V^-1 b (statsmodels b, V)", o["Chi2"], float(b @ np.linalg.solve(V, b)), 1e-8)
        rec(f"S5.4 {feat}: ICC by hand DT/(DT+DN)", o["ICC"], o["DT"] / (o["DT"] + o["DN"]), 1e-12)
        rec(f"S5.4 {feat}: Trial_CV% by hand sqrt(DN)/mean", o["Trial_CV%"], 100 * np.sqrt(o["DN"]) / sdf[feat].mean(), 1e-12)
    rec("S5.4: Holm (m = 5) by hand, largest adjusted p", R["omni"]["p_holm"].max(),
        verify.holm(R["omni"]["p_unc"]).max(), 1e-9)

    vdfs = R["val_full"]["dfs"]
    for feat in FEATURES:
        v = R["val"].loc[feat]
        fit = verify.ri_lme(f"{feat}_rec ~ {feat}_targ", vdfs[feat], "participant_code")
        rec(f"S5.3 {feat}: slope b, independent ML fit", v["b"], fit["beta"][f"{feat}_targ"], 1e-4)
        rec(f"S5.3 {feat}: z", v["z"], fit["beta"][f"{feat}_targ"] / fit["se"][f"{feat}_targ"], 2e-3,
            "statsmodels SEs come from the full Hessian; with participant variance at the 0 boundary "
            "they differ slightly from the GLS formula (X'V^-1 X)^-1")
        rec(f"S5.3 {feat}: residual variance", v["DN"], fit["sigma2"], 1e-3)
        rows.append({"check": f"S5.3 {feat}: participant variance (boundary cases ~0)",
                     "statsmodels / pipeline": v["DT"], "independent": fit["tau2"],
                     "rel_diff": abs(v["DT"] - fit["tau2"]), "tolerance": 1e-5,
                     "status": "PASS" if abs(v["DT"] - fit["tau2"]) <= 1e-5 else "FAIL",
                     "note": "absolute difference (variance at or near 0)"})
    rec("S5.3: Holm (m = 5) by hand, largest adjusted p", R["val"]["p_holm"].max(),
        verify.holm(R["val"]["p_unc"]).max(), 1e-9)

    d1015 = sdf[sdf["condition_hz"].isin([10, 15])].copy()
    d1015["is15"] = (d1015["condition_hz"] == 15).astype(float)
    fit = verify.ri_lme("detail ~ is15", d1015, "participant_code")
    c = R["con"]
    row = c[(c["feature"] == "detail") & (c["Contrast"] == "15 Hz vs 10 Hz")].iloc[0]
    rec("10 vs 15 Hz detail: b, independent ML fit", row["b"], fit["beta"]["is15"], 1e-4)
    rec("10 vs 15 Hz detail: SE", row["se"], fit["se"]["is15"], 1e-3)
    rec("10 vs 15 Hz detail: z", row["z"], fit["beta"]["is15"] / fit["se"]["is15"], 1e-3)

    o = R["order"]
    for fam in ("Experimental", "Validation"):
        sub = o[o["Trials"] == fam]
        adj = verify.holm(np.r_[sub["p_trial"], sub["p_session"]])
        rec(f"S5.2 {fam}: Holm (m = 10) by hand, sum of adjusted p",
            float(np.r_[sub["p_trial_holm"], sub["p_session_holm"]].sum()), float(adj.sum()), 1e-9)
        rec(f"S5.2 {fam}: drift = 15 x b (detail)", float(sub["drift"].iloc[0]), 15 * float(sub["b"].iloc[0]), 1e-12)
        pct = sub[["pct_main", "pct_trial", "pct_session", "pct_participant", "pct_resid"]].sum(axis=1)
        rec(f"S5.2 {fam}: variance shares sum to 100% (worst row)", 100.0,
            float(pct.iloc[int((pct - 100).abs().to_numpy().argmax())]), 1e-9)

    ct = R["content"]["crosstab"].to_numpy(float)
    E = np.outer(ct.sum(1), ct.sum(0)) / ct.sum()
    rec("5.3.2.1 chi-square by hand sum((O-E)^2/E)", R["content"]["chi2"], float(((ct - E) ** 2 / E).sum()), 1e-12)
    rows.append({"check": "5.3.2.1 chi-square Monte Carlo p (20,000 permutations, seed 42)",
                 "statsmodels / pipeline": R["content"]["p"], "independent": R["content"]["mc_p"],
                 "rel_diff": np.nan, "tolerance": np.nan,
                 "status": "PASS" if R["content"]["mc_p"] < 0.001 else "FAIL",
                 "note": f"{100 * R['content']['expected_lt5_frac']:.0f}% of expected counts < 5 "
                         f"(min {R['content']['expected_min']:.2f}); asymptotic p is approximate"})

    # optimiser sensitivity: what the notebook rule would pick in this environment
    for feat in FEATURES:
        md = __import__("statsmodels.formula.api", fromlist=["mixedlm"]).mixedlm(
            f"{feat} ~ C(condition_hz)", sdf, groups=sdf["participant_code"])
        for method in ("lbfgs", "bfgs", "cg", "powell", "nm"):
            try:
                r = md.fit(reml=False, method=method, maxiter=2000, disp=False)
                rows.append({"check": f"optimiser {method}: S5.4 {feat}", "statsmodels / pipeline": r.llf,
                             "independent": float(r.cov_re.iloc[0, 0]), "rel_diff": np.nan, "tolerance": np.nan,
                             "status": "INFO", "note": f"converged={r.converged}; columns = llf, participant variance"})
            except Exception as e:  # noqa: BLE001
                rows.append({"check": f"optimiser {method}: S5.4 {feat}", "statsmodels / pipeline": np.nan,
                             "independent": np.nan, "rel_diff": np.nan, "tolerance": np.nan,
                             "status": "INFO", "note": f"raised {type(e).__name__}: {e}"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- optional steps
def run_notebook(path: Path, timeout: int = 3600, save: bool = False):
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    nb = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(path.parent)}})
    if save:
        nbformat.write(nb, path)


def with_images(checks: Checks):
    img_dir = PROJECT_PATH / "005_cleaned_data"
    if not img_dir.is_dir():
        checks.hard("--with-images: 005_cleaned_data/ present", False, True,
                    "copy the cleaned images into 005_cleaned_data/ first")
        checks.abort_if_failed("with-images")
    before = pd.read_csv(TABLES_PATH / "vizex_features.csv")
    nbdir = PROJECT_PATH / "011_notebooks_v2"
    for nb in ("010_build_index.ipynb", "015_feature_extraction.ipynb"):
        print(f"executing {nb} on the images (this can take a while) ...", flush=True)
        run_notebook(nbdir / nb)
    after = pd.read_csv(TABLES_PATH / "vizex_features.csv")
    same_rows = set(before["png_filename"]) == set(after["png_filename"])
    checks.hard("--with-images: image pipeline gives the same 1,580 trials", same_rows, True)
    if same_rows:
        m = before.merge(after, on="png_filename", suffixes=("_csv", "_img"))
        diff = max(float((m[f + "_csv"] - m[f + "_img"]).abs().max()) for f in FEATURES)
        checks.soft("--with-images: features from images == CSV-derived features", diff < 1e-6,
                    f"max abs diff {diff:.2g}", "< 1e-6")
    dest = FIG_PATH / "S5.4_heatmaps"
    dest.mkdir(parents=True, exist_ok=True)
    for p in sorted((PROJECT_PATH / "020_visualizations").glob("hm_vizex_*detail*.png")):
        shutil.copy2(p, dest / p.name)
    checks.info("--with-images: Figure S5.4 detail heatmaps copied", str(dest.relative_to(PROJECT_PATH)))


def notebooks_check(checks: Checks, R: dict):
    nbdir = PROJECT_PATH / "011_notebooks_v2"
    for nb in ("030_feature_validation_analysis.ipynb", "035_feature_experimental_analysis.ipynb",
               "036_supplementary_control_analyses.ipynb"):
        print(f"executing {nb} ...", flush=True)
        run_notebook(nbdir / nb, save=True)
    comps = [
        ("030 validation (b, se, DT, DN)", "table_validation_raw.csv", R["val_full"]["raw"], ["b", "se", "DT", "DN"], "Feature"),
        ("035 omnibus (Chi2, DT, DN)", "table_experimental_raw.csv", R["exp_full"]["omnibus"],
         ["Chi2", "DT_plus_HT", "DN_plus_HN"], "Feature"),
        ("036 trial order (b, chi2_trial, pct_session)", "table_S5.3_raw.csv", R["order"],
         ["b", "chi2_trial", "pct_session"], ["Trials", "Feature"]),
    ]
    for name, fn, ours, cols, key in comps:
        nbt = pd.read_csv(TABLES_PATH / fn)
        ours = ours.rename(columns={"DT": "DT_plus_HT", "DN": "DN_plus_HN"}) if "DT_plus_HT" in cols else ours
        m = nbt.merge(ours, on=key, suffixes=("_nb", "_py"))
        # chi-square statistics printed to 1-2 dp: optimiser tolerance (~1e-4) is irrelevant
        tol = {c: (1e-3 if c.lower().startswith("chi2") else 1e-6) for c in cols}
        bad = {c: float((m[c + "_nb"] - m[c + "_py"]).abs().max()) for c in cols}
        ok = all(bad[c] <= tol[c] for c in cols) and len(m) == len(ours)
        checks.soft(f"--notebooks: {name} notebook == script", ok,
                    ", ".join(f"{c} max |diff| {v:.2g}" for c, v in bad.items()), "chi2 <= 1e-3, others <= 1e-6")


# ----------------------------------------------------------------- main
def baseline_data(checks: Checks):
    try:
        def rd(p):
            out = subprocess.run(["git", "-C", str(PROJECT_PATH), "show", f"{BASELINE_COMMIT}:015_tables/{p}"],
                                 capture_output=True, text=True, check=True).stdout
            return pd.read_csv(io.StringIO(out), dtype=str, keep_default_na=False)
        return build_tables.build(checks, write=False, apply_manual=False, trials_master=rd("trials_master.csv"),
                                  features=rd("vizex_features.csv"), targets=rd("vizex_targets_features.csv"),
                                  label="pre-exclusion")
    except Exception as e:  # noqa: BLE001
        checks.info("pre-exclusion reproduction", f"skipped ({type(e).__name__}: {e})")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--with-images", action="store_true")
    ap.add_argument("--notebooks", action="store_true")
    ap.add_argument("--mc-sims", type=int, default=20000)
    args = ap.parse_args()

    t0 = time.time()
    set_seeds()
    quiet_warnings()
    ensure_dirs()
    checks = Checks()
    import matplotlib, scipy, statsmodels  # noqa: E401
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
                "matplotlib": matplotlib.__version__}
    banner("Environment: " + ", ".join(f"{k} {v}" for k, v in versions.items()))

    if args.with_images:
        banner("Optional: 010 + 015 on the images")
        with_images(checks)

    banner("Step 1: apply manual exclusions and rebuild the analysis tables (015_tables/)")
    data = build_tables.build(checks, write=True, label="new")
    hard_counts(checks, data, "new")
    checks.abort_if_failed("analysis-set counts")

    banner("Step 2: Chapter 5 analyses on the corrected data (1,580 trials)")
    R_new = analyse(data, args.mc_sims)
    model_counts(checks, R_new, "new")
    checks.abort_if_failed("model counts")
    write_outputs(R_new)

    R_base = None
    if not args.no_baseline:
        banner("Step 3: the same code on the pre-exclusion data (reproduces the thesis)")
        base = baseline_data(checks)
        if base is not None:
            R_base = analyse(base, 0)
            checks.hard("[pre-exclusion] SLS trials in experimental models", int(R_base["omni"]["n"].iloc[0]), 1285)

    banner("Step 4: independent verification (part D)")
    ver = verification(checks, R_new, data)
    ver.to_csv(OUT_PATH / "VERIFICATION.csv", index=False)

    if args.notebooks:
        banner("Optional: execute notebooks 030/035/036 and compare")
        notebooks_check(checks, R_new)

    banner("Step 5: OLD_VS_NEW.md")
    rows = report.build_rows(thesis_values.items(), R_new, R_base)
    stable = [
        ("10 vs 15 Hz follow-up (b, SE, z) unchanged at printed precision",
         rows[rows["Location"].str.contains("follow-up") & rows["Quantity"].str.contains(r": (b|SE|z)$")]),
        ("validation slopes, SEs, z and p (Table S5.3) unchanged",
         rows[rows["Location"].str.startswith("Table S5.3") & rows["Quantity"].isin(["n", "b (slope)", "SE", "z", "p (Holm, m = 5)"])]),
    ]
    for name, sub in stable:
        changed = sub[sub["Status"] != "unchanged"]
        checks.soft(name, changed.empty, f"{len(sub) - len(changed)}/{len(sub)} unchanged",
                    "all unchanged", "; ".join(f"{r.Location} {r.Quantity}: {r.New}" for r in changed.itertuples()))
    preamble = (
        "Thesis Chapter 5 numbers (HEWITT_THESIS_FULL_9.docx) against the rerun with P358 trial 8 "
        "excluded (1,580 trials: 1,186 experimental, 98 control, 296 validation; 99 participants). "
        "`Pre-exclusion reproduction` runs the identical code on the 1,581-trial tables at commit "
        f"`{BASELINE_COMMIT[:7]}`; it shows whether a difference comes from the exclusion or was already "
        "there. The edge features are re-normalised to the 0.95 quantile of the 1,580 in-analysis "
        "trials, as 015_feature_extraction does (scale ratio new/old: "
        + ", ".join(f"{k} {R_new['scales'][k] / R_base['scales'][k]:.6f}" for k in ("detail", "r_directionality", "theta_directionality"))
        if R_base is not None else "Thesis Chapter 5 numbers against the rerun (pre-exclusion reproduction skipped).")
    if R_base is not None:
        preamble += ")."
    report.write(rows, OUT_PATH / "OLD_VS_NEW.md", "Chapter 5: old (thesis) vs new (1,580 trials)", preamble)

    (OUT_PATH / "RUN_INFO.md").write_text(
        "# Run information\n\n" + "\n".join(f"* {k}: {v}" for k, v in versions.items())
        + f"\n* options: {vars(args)}\n* pre-exclusion commit: {BASELINE_COMMIT}\n"
        + f"* checks: {checks.summary()}\n", encoding="utf-8")
    checks.write()
    banner(f"Done in {time.time() - t0:.0f} s. Checks: {checks.summary()}. "
           f"Send back the whole rerun_outputs/ folder.")
    hard_failed = any(r["kind"] == "hard" and r["status"] == "FAIL" for r in checks.rows)
    sys.exit(1 if hard_failed else 0)


if __name__ == "__main__":
    main()
