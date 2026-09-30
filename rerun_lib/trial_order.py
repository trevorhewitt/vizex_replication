"""Trial order and session (thesis Supplementary S5.4, Table S5.2) and the
recreation-accuracy-by-trial check (computed in the notebook, not reported in
the thesis).

Port of 011_notebooks_v2/036_supplementary_control_analyses.ipynb as of
6e52ac6 / b1ec73a (the version that reproduces thesis Table S5.2):
* the three December 2025 trials (trialN 101/102) are excluded here only;
* P368's in-session make-up (trialN 1) sits midway between the trials saved
  either side of it (position 4.5);
* features are z-scored within the experimental and within the validation data;
* feature ~ condition + trial_c with nested random intercepts
  (participant within session); trial tested by LRT (1 df); session by LRT
  against a participant-only intercept, halved p for the boundary;
* Holm within each family of ten (5 trial + 5 session tests);
* drift = b x (max position - min position) = b x 15;
* % variance = share of (fixed main + fixed trial + session + participant +
  residual) variance.
The best of several optimisers (highest log-likelihood) is kept.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from .common import FEATURE_LABELS as FEATURES

MAKEUP_TRIALN = [1]
MAX_TRIALN = 17
OPTIMISERS = ("bfgs", "powell", "cg", "nm")
VC_NESTED = {"participant": "0 + C(participant_code)"}
FNAME_PATTERN = (
    r"sessionTime_(?P<session>\d+)"
    r".*?participant_(?P<pid_file>[A-Za-z0-9]+)"
    r".*?trialN_(?P<trialn_file>\d+)"
    r".*?trialT_(?P<trialt_file>[A-Za-z0-9]+)"
    r"__fileType_(?P<file_type>[A-Za-z0-9]+)\.png"
)


def prepare(df_vizex: pd.DataFrame, df_targets: pd.DataFrame) -> dict:
    meta = df_vizex["png_filename"].astype(str).str.extract(FNAME_PATTERN)
    assert meta["session"].notna().all(), "Filename pattern failed."
    d = pd.concat([df_vizex.reset_index(drop=True), meta], axis=1)
    d = d[d["file_type"] == "cropped"]
    n_late = int((d["trialn"].astype(int) > MAX_TRIALN).sum())
    d = d[d["trialn"].astype(int) <= MAX_TRIALN].copy()
    d["session"] = d["session"].astype(str)
    d["condition_code"] = d["condition_code"].astype(str)
    makeup = d["trialn"].astype(int).isin(MAKEUP_TRIALN)
    first_trialn = int(d.loc[~makeup, "trialn"].astype(int).min())
    d["trial_pos"] = (d["trialn"].astype(int) - first_trialn + 1).astype(float)
    placed = []
    for i in d.index[makeup]:
        own = d[(d["participant_code"] == d.at[i, "participant_code"]) & ~makeup]
        t = d.at[i, "save_time"]
        before = own.loc[own["save_time"] < t].sort_values("save_time")["trial_pos"]
        after = own.loc[own["save_time"] > t].sort_values("save_time")["trial_pos"]
        assert len(before) and len(after), "make-up trial not bracketed by in-session trials"
        d.at[i, "trial_pos"] = (before.iloc[-1] + after.iloc[0]) / 2
        placed.append((d.at[i, "participant_code"], int(d.at[i, "trialn"]),
                       d.at[i, "condition_code"], float(d.at[i, "trial_pos"])))
    trial_span = int(d["trial_pos"].max() - d["trial_pos"].min())
    d["trial_c"] = d["trial_pos"] - d["trial_pos"].mean()

    exp = d[d["stimulus_type"] == "LIGHT"].copy()
    exp["condition_hz"] = pd.to_numeric(exp["condition_hz"], errors="coerce").round(2)
    for f in FEATURES:
        exp[f] = pd.to_numeric(exp[f], errors="coerce")
    exp = exp.dropna(subset=list(FEATURES) + ["condition_hz", "session", "participant_code"]).copy()
    for f in FEATURES:
        exp["z_" + f] = (exp[f] - exp[f].mean()) / exp[f].std(ddof=1)

    tg = df_targets.copy()
    tg["condition_code"] = tg["condition_code"].astype(str)
    tg = tg[["condition_code"] + list(FEATURES)].rename(columns={f: "stim_" + f for f in FEATURES})
    val = d[(d["stimulus_type"] != "LIGHT") & (d["condition_code"].isin(set(tg["condition_code"])))].copy()
    val = val.merge(tg, on="condition_code", how="left", validate="many_to_one")
    for f in FEATURES:
        val[f] = pd.to_numeric(val[f], errors="coerce")
        val["stim_" + f] = pd.to_numeric(val["stim_" + f], errors="coerce")
    val = val.dropna(subset=list(FEATURES) + ["stim_" + f for f in FEATURES]).copy()
    for f in FEATURES:
        m, s = val[f].mean(), val[f].std(ddof=1)
        val["z_" + f] = (val[f] - m) / s
        val["zs_" + f] = (val["stim_" + f] - val["stim_" + f].mean()) / val["stim_" + f].std(ddof=1)
        val["err_" + f] = ((val[f] - val["stim_" + f]) / s).abs()

    # design checks, as in the notebook
    assert (d["pid_file"] == d["participant_code"]).all(), "filename participant mismatch"
    assert (d["trialn_file"].astype(int) == d["trialn"].astype(int)).all(), "filename trial mismatch"
    assert d.duplicated(["participant_code", "trialn"]).sum() == 0, "duplicate participant x trial rows"
    assert (d.groupby("participant_code")["session"].nunique() == 1).all(), "participant spans sessions"
    r2 = smf.ols("trial_c ~ C(condition_hz)", exp).fit().rsquared
    return {"d": d, "exp": exp, "val": val, "trial_span": trial_span, "n_late": n_late,
            "placed": placed, "r2_position_on_frequency": r2,
            "n_sessions": int(d["session"].nunique())}


def fit_best(formula, data, groups, vc_formula=None):
    model = smf.mixedlm(formula, data, groups=data[groups], re_formula="1", vc_formula=vc_formula)
    best, err = None, None
    for method in OPTIMISERS:
        try:
            res = model.fit(reml=False, method=method)
            if np.isfinite(res.llf) and (best is None or res.llf > best.llf):
                best = res
        except Exception as e:  # noqa: BLE001 - as in the notebook
            err = e
    if best is None:
        raise RuntimeError(f"all optimisers failed for '{formula}': {err}")
    return model, best


def lrt(res_full, res_reduced, df, boundary=False):
    stat = max(2 * (res_full.llf - res_reduced.llf), 0.0)
    p = chi2.sf(stat, df)
    return stat, (0.5 * p if boundary and stat > 0 else (1.0 if boundary else p))


def fixed_var(model, res, columns):
    names = list(model.exog_names)
    idx = [names.index(c) for c in columns if c in names]
    return float(np.var(np.asarray(model.exog)[:, idx] @ np.asarray(res.fe_params)[idx])) if idx else 0.0


def accuracy(val: pd.DataFrame) -> pd.DataFrame:
    val_acc = val[val["trial_pos"] % 1 == 0]
    rows = []
    for feat, name in FEATURES.items():
        y = "err_" + feat
        model, res = fit_best(f"{y} ~ C(trial_pos)", val_acc, "participant_code")
        _, res0 = fit_best(f"{y} ~ 1", val_acc, "participant_code")
        df_trial = len([c for c in model.exog_names if c.startswith("C(trial_pos)")])
        stat, p = lrt(res, res0, df_trial)
        means = np.asarray(model.exog) @ np.asarray(res.fe_params)
        rows.append({"Feature": name, "mean_err": val_acc[y].mean(),
                     "spread": float(np.max(means) - np.min(means)),
                     "chi2": stat, "df": df_trial, "p": p, "n": len(val_acc)})
    acc = pd.DataFrame(rows)
    acc["p_holm"] = multipletests(acc["p"], alpha=0.05, method="holm")[1]
    return acc


def run_family(data, template, main_prefix, label, trial_span):
    rows = []
    for feat, name in FEATURES.items():
        formula = template.format(f=feat)
        model, res = fit_best(formula, data, "session", VC_NESTED)
        _, res_notrial = fit_best(formula.replace(" + trial_c", ""), data, "session", VC_NESTED)
        _, res_nosession = fit_best(formula, data, "participant_code")
        main_cols = [c for c in model.exog_names if c.startswith(main_prefix)]
        v = {"main": fixed_var(model, res, main_cols),
             "trial": fixed_var(model, res, ["trial_c"]),
             "session": float(np.asarray(res.cov_re)[0, 0]),
             "participant": float(np.atleast_1d(res.vcomp)[0]),
             "resid": float(res.scale)}
        total = sum(v.values())
        chi_t, p_t = lrt(res, res_notrial, 1)
        chi_s, p_s = lrt(res, res_nosession, 1, boundary=True)
        b = float(res.fe_params["trial_c"])
        try:
            se = float(res.bse_fe["trial_c"])
        except Exception:  # noqa: BLE001
            se = np.nan
        rows.append({"Trials": label, "Feature": name, "n": len(data), "b": b, "se": se,
                     "chi2_trial": chi_t, "p_trial": p_t, "drift": b * trial_span,
                     "chi2_session": chi_s, "p_session": p_s,
                     **{f"pct_{k}": 100 * x / total for k, x in v.items()}})
    out = pd.DataFrame(rows)
    p_all = multipletests(np.r_[out["p_trial"], out["p_session"]], alpha=0.05, method="holm")[1]
    out["p_trial_holm"] = p_all[:len(out)]
    out["p_session_holm"] = p_all[len(out):]
    return out


def coefficient_shift(exp):
    shifts, corrs = [], []
    for feat in FEATURES:
        m_b, r_b = fit_best(f"z_{feat} ~ C(condition_hz)", exp, "participant_code")
        m_f, r_f = fit_best(f"z_{feat} ~ C(condition_hz) + trial_c", exp, "session", VC_NESTED)
        cols = [c for c in m_b.exog_names if c.startswith("C(condition_hz)")]
        shifts.append(np.max(np.abs(r_f.fe_params[cols].values - r_b.fe_params[cols].values)))
        corrs.append(np.corrcoef(r_b.fe_params[cols].values, r_f.fe_params[cols].values)[0, 1])
    return float(np.max(shifts)), float(np.min(corrs))


def _fmt_p(p):
    if not np.isfinite(p):
        return "-"
    return "< .001" if p < 0.001 else f"{p:.3f}".replace("0.", ".", 1)


def paper_table(order_res: pd.DataFrame) -> pd.DataFrame:
    """Thesis Table S5.2 layout (the notebook calls it S5.3)."""
    return pd.DataFrame({
        "Trials": order_res["Trials"].map({"Experimental": "Exp.", "Validation": "Val."}),
        "Feature": order_res["Feature"],
        "b per trial (SE)": [f"{b:+.4f} (-)" if not np.isfinite(se) else f"{b:+.4f} ({se:.4f})"
                             for b, se in zip(order_res["b"], order_res["se"])],
        "chi2(1)": order_res["chi2_trial"].map("{:.2f}".format),
        "p": [_fmt_p(v) for v in order_res["p_trial_holm"]],
        "Drift across experiment (SD)": order_res["drift"].map("{:+.3f}".format),
        "Trial order (% variance)": order_res["pct_trial"].map("{:.2f}".format),
        "Session chi2(1)": order_res["chi2_session"].map("{:.2f}".format),
        "Session p": [_fmt_p(v) for v in order_res["p_session_holm"]],
        "Session (% variance)": order_res["pct_session"].map("{:.2f}".format),
    })


def run(df_vizex: pd.DataFrame, df_targets: pd.DataFrame) -> dict:
    prep = prepare(df_vizex, df_targets)
    exp, val, span = prep["exp"], prep["val"], prep["trial_span"]
    acc = accuracy(val)
    ord_exp = run_family(exp, "z_{f} ~ C(condition_hz) + trial_c", "C(condition_hz)", "Experimental", span)
    ord_val = run_family(val, "z_{f} ~ zs_{f} + trial_c", "zs_", "Validation", span)
    order = pd.concat([ord_exp, ord_val], ignore_index=True)
    max_shift, min_r = coefficient_shift(exp)
    return {**prep, "accuracy": acc, "order": order, "max_shift": max_shift, "min_r": min_r}
