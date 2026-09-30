"""Every Chapter 5 number the thesis reports from these analyses, with its
location, as printed in HEWITT_THESIS_FULL_9.docx (the "old" column of
OLD_VS_NEW.md), and how to compute the same quantity from the rerun.

Each item: (location, quantity, thesis text, getter, formatter, kind)
kind: 'num' (compare formatted text), 'p' (also compare the p < .05 verdict),
      'claim' (getter returns (bool, evidence text); thesis text is the claim).
Getters take the results dict R assembled in rerun.py.
"""

from .common import FEATURES, format_p

FEAT_ROWS = {  # thesis row label -> feature column
    "detail level": "detail", "radial orientation": "r_directionality",
    "tangential orientation": "theta_directionality", "brightness": "brightness",
    "contrast": "contrast",
}
S52_LABEL = {"detail": "detail level", "r_directionality": "radial",
             "theta_directionality": "tangential", "brightness": "brightness", "contrast": "contrast"}

f6 = lambda v: f"{v:.6f}"
f4 = lambda v: f"{v:.4f}"
f3 = lambda v: f"{v:.3f}"
f2 = lambda v: f"{v:.2f}"
f1 = lambda v: f"{v:.1f}"
f0 = lambda v: f"{v:,.0f}"
pct0 = lambda v: f"{v:.0f}%"
pct1 = lambda v: f"{v:.1f}%"
fint = lambda v: f"{int(v):,}"
fp = format_p


def _v(R, feat, col):
    return R["val"].loc[feat, col]


def _o(R, feat, col):
    return R["omni"].loc[feat, col]


def _t(R, trials, feat, col):
    return R["order"].set_index(["Trials", "feature"]).loc[(trials, feat), col]


def _c(R, feat, contrast, col):
    c = R["con"]
    return c[(c["feature"] == feat) & (c["Contrast"] == contrast)][col].iloc[0]


def _m(R, feat):
    return R["means"].set_index("condition_hz")[f"{feat}_mean"]


def items():
    L = []
    add = lambda *a: L.append(a)

    # ------------------------------------------------------------ counts
    add("Title; 5.2.1.4; 5.3", "trials analysed", "1,580", lambda R: R["counts"]["in_analysis"], fint, "num")
    add("5.2.1.4", "experimental trials", "1,186", lambda R: R["counts"]["experimental"], fint, "num")
    add("5.2.1.4", "control trials", "98", lambda R: R["counts"]["control"], fint, "num")
    add("5.2.1.4", "validation trials", "296", lambda R: R["counts"]["validation"], fint, "num")
    add("5.2.1; 5.3; Abstract", "participants", "99", lambda R: R["counts"]["participants"], fint, "num")

    # ------------------------------------------------------------ Table S5.3
    s53 = {
        "detail level": ("296", "0.427464", "0.022199", "19.26", "< .001", "0.019314", "0.064122"),
        "radial orientation": ("296", "0.753533", "0.014673", "51.35", "< .001", "0.000000", "0.010990"),
        "tangential orientation": ("296", "0.859217", "0.017720", "48.49", "< .001", "0.000000", "0.008976"),
        "brightness": ("296", "0.931987", "0.037206", "25.05", "< .001", "0.002225", "0.015500"),
        "contrast": ("296", "0.840334", "0.035239", "23.85", "< .001", "0.000422", "0.003758"),
    }
    for row, vals in s53.items():
        f = FEAT_ROWS[row]
        loc = f"Table S5.3, {row}"
        add(loc, "n", vals[0], lambda R, f=f: _v(R, f, "n"), fint, "num")
        add(loc, "b (slope)", vals[1], lambda R, f=f: _v(R, f, "b"), f6, "num")
        add(loc, "SE", vals[2], lambda R, f=f: _v(R, f, "se"), f6, "num")
        add(loc, "z", vals[3], lambda R, f=f: _v(R, f, "z"), f2, "num")
        add(loc, "p (Holm, m = 5)", vals[4], lambda R, f=f: _v(R, f, "p_holm"), fp, "p")
        add(loc, "random-intercept variance", vals[5], lambda R, f=f: _v(R, f, "DT"), f6, "num")
        add(loc, "residual variance", vals[6], lambda R, f=f: _v(R, f, "DN"), f6, "num")
    add("5.3.1; Fig 5.1 caption", "smallest validation slope (detail level)", "0.43",
        lambda R: _v(R, "detail", "b"), f2, "num")
    add("5.3.1; Fig 5.1 caption", "largest validation slope (brightness)", "0.93",
        lambda R: _v(R, "brightness", "b"), f2, "num")
    add("5.3.1", "average validation slope", "0.76", lambda R: R["val"]["b"].mean(), f2, "num")
    add("5.3.1", "all validation tests p < .001", "p < .001 (all)",
        lambda R: (bool((R["val"]["p_holm"] < 0.001).all()),
                   f"max Holm p = {R['val']['p_holm'].max():.2e}"), None, "claim")
    add("5.4.1", "validation ICC range (from Table S5.3)", "0 to 0.23",
        lambda R: f"{_g(R['val']['ICC'].min())} to {_g(R['val']['ICC'].max())}", str, "num")

    # ------------------------------------------------------------ Table S5.4
    s54 = {
        "detail level": ("1284", "0.16", "130.2", "6", "< .001", "0.19", "116.28%", "0.406", "0.023849", "0.034949"),
        "radial orientation": ("1284", "0.40", "126.5", "6", "< .001", "0.24", "59.37%", "0.158", "0.010817", "0.057817"),
        "tangential orientation": ("1284", "0.44", "117.6", "6", "< .001", "0.24", "55.56%", "0.137", "0.009349", "0.059029"),
        "brightness": ("1284", "0.78", "113.4", "6", "< .001", "0.15", "19.61%", "0.294", "0.009658", "0.023215"),
        "contrast": ("1284", "0.14", "125.1", "6", "< .001", "0.09", "61.90%", "0.390", "0.004674", "0.007302"),
    }
    for row, vals in s54.items():
        f = FEAT_ROWS[row]
        loc = f"Table S5.4, {row}"
        add(loc, "n", vals[0], lambda R, f=f: _o(R, f, "n"), lambda v: str(int(v)), "num")
        add(loc, "Mean", vals[1], lambda R, f=f: _o(R, f, "Mean"), f2, "num")
        add(loc, "Chi2 (Wald)", vals[2], lambda R, f=f: _o(R, f, "Chi2"), f1, "num")
        add(loc, "df", vals[3], lambda R, f=f: _o(R, f, "df"), lambda v: str(int(v)), "num")
        add(loc, "p (Holm, m = 5)", vals[4], lambda R, f=f: _o(R, f, "p_holm"), fp, "p")
        add(loc, "Trial SD", vals[5], lambda R, f=f: _o(R, f, "Trial_SD"), f2, "num")
        add(loc, "Trial_CV%", vals[6], lambda R, f=f: _o(R, f, "Trial_CV%"), lambda v: f"{v:.2f}%", "num")
        add(loc, "ICC", vals[7], lambda R, f=f: _o(R, f, "ICC"), f3, "num")
        add(loc, "random-intercept variance", vals[8], lambda R, f=f: _o(R, f, "DT"), f6, "num")
        add(loc, "residual variance", vals[9], lambda R, f=f: _o(R, f, "DN"), f6, "num")

    # ------------------------------------------------------------ 5.3.2.2 text
    add("5.3.2.2", "smallest omnibus Wald chi2(6) ('> 113')", "> 113 (smallest 113.4, brightness)",
        lambda R: (bool(R["omni"]["Chi2"].min() > 113),
                   f"smallest = {R['omni']['Chi2'].min():.1f} ({R['omni']['Chi2'].idxmin()})"), None, "claim")
    add("5.3.2.2", "all five omnibus p < .001", "p < .001 for all five",
        lambda R: (bool((R["omni"]["p_holm"] < 0.001).all()),
                   f"max Holm p = {R['omni']['p_holm'].max():.1e}"), None, "claim")
    add("5.3.2.2", "detail level peaks at 15 Hz", "peak at 15 Hz",
        lambda R: (_m(R, "detail").idxmax() == 15, "means: " + _means_txt(R, "detail")), None, "claim")
    add("5.3.2.2", "tangential orientation minimum at 15 Hz", "minimum at 15 Hz",
        lambda R: (_m(R, "theta_directionality").idxmin() == 15,
                   "means: " + _means_txt(R, "theta_directionality")), None, "claim")
    add("5.3.2.2", "tangential orientation mean at 80 Hz", "0.62",
        lambda R: _m(R, "theta_directionality").loc[80.0], f2, "num")
    add("5.3.2.2", "80 Hz has the highest tangential mean", "highest of any condition",
        lambda R: (_m(R, "theta_directionality").idxmax() == 80, ""), None, "claim")
    add("5.3.2.2", "radial orientation 'similar, less pronounced' inverted U (peak location)",
        "similar to detail (peak 15 Hz implied)",
        lambda R: (_m(R, "r_directionality").idxmax() == 15, "means: " + _means_txt(R, "r_directionality")),
        None, "claim")
    add("5.3.2.2", "brightness increases with frequency (strictly monotonic?)", "increased with frequency",
        lambda R: (bool(_m(R, "brightness").is_monotonic_increasing), "means: " + _means_txt(R, "brightness")),
        None, "claim")
    add("5.3.2.2", "contrast decreases with frequency (strictly monotonic?)", "decreased",
        lambda R: (bool(_m(R, "contrast").is_monotonic_decreasing), "means: " + _means_txt(R, "contrast")),
        None, "claim")
    add("5.3.2.2", "above 15 Hz: brighter, less detail, less radial, lower contrast (15>20>40>80)",
        "monotonic from 15 to 80 Hz",
        lambda R: (_beyond15(R), ""), None, "claim")
    add("5.3.2.2 follow-up", "10 vs 15 Hz detail: b", "0.071", lambda R: _c(R, "detail", "15 Hz vs 10 Hz", "b"), f3, "num")
    add("5.3.2.2 follow-up", "10 vs 15 Hz detail: SE", "0.021", lambda R: _c(R, "detail", "15 Hz vs 10 Hz", "se"), f3, "num")
    add("5.3.2.2 follow-up", "10 vs 15 Hz detail: z", "3.39", lambda R: _c(R, "detail", "15 Hz vs 10 Hz", "z"), f2, "num")
    add("5.3.2.2 follow-up", "10 vs 15 Hz detail: p (single test, uncorrected)", "< .001",
        lambda R: _c(R, "detail", "15 Hz vs 10 Hz", "p_unc"), fp, "p")
    add("5.3.2.2 follow-up", "10 vs 15 Hz detail: n trials in model (not printed)", "(395 = 198 + 197)",
        lambda R: _c(R, "detail", "15 Hz vs 10 Hz", "n"), fint, "info")

    # ------------------------------------------------------------ 5.3.2.3 / 5.4.1
    add("5.3.2.3", "detail residual SD", "0.19", lambda R: _o(R, "detail", "Trial_SD"), f2, "num")
    add("5.3.2.3", "detail grand mean", "0.16", lambda R: _o(R, "detail", "Mean"), f2, "num")
    add("5.3.2.3", "detail residual SD exceeds grand mean", "0.19 > 0.16",
        lambda R: (bool(_o(R, "detail", "Trial_SD") > _o(R, "detail", "Mean")), ""), None, "claim")
    add("5.3.2.3", "brightness residual SD / mean ('one-fifth')", "19.61%",
        lambda R: _o(R, "brightness", "Trial_CV%"), lambda v: f"{v:.2f}%", "num")
    add("5.3.2.3", "contrast residual SD / mean ('about three-fifths')", "61.90%",
        lambda R: _o(R, "contrast", "Trial_CV%"), lambda v: f"{v:.2f}%", "num")
    for name, f, txt in (("radial", "r_directionality", "0.16"), ("tangential", "theta_directionality", "0.14"),
                         ("detail level", "detail", "0.41"), ("contrast", "contrast", "0.39"),
                         ("brightness", "brightness", "0.29")):
        add("5.3.2.3", f"ICC {name}", txt, lambda R, f=f: _o(R, f, "ICC"), f2, "num")
    add("5.4.1", "ICC range (% of unexplained variance)", "14% to 41%",
        lambda R: f"{100 * R['omni']['ICC'].min():.0f}% to {100 * R['omni']['ICC'].max():.0f}%", str, "num")
    add("6.1.4.3", "ICC up to", "0.41", lambda R: R["omni"]["ICC"].max(), f2, "num")

    # ------------------------------------------------------------ content (5.3.2.1)
    add("5.3.2.1", "2.5 Hz: geometric content present", "61%", lambda R: R["content"]["pct_geo_2_5"], pct0, "num")
    add("5.3.2.1", "2.5 Hz: Null", "39%", lambda R: R["content"]["pct_null_2_5"], pct0, "num")
    add("5.3.2.1", "5-40 Hz: geometric content present", "96%", lambda R: R["content"]["pct_geo_5_40"], pct0, "num")
    add("5.3.2.1; 5.3.2.2", "80 Hz: Null", "82%", lambda R: R["content"]["pct_null_80"], pct0, "num")
    add("5.3.2.1", "chi-square (18 df)", "548.76", lambda R: R["content"]["chi2"], f2, "num")
    add("5.3.2.1", "chi-square df", "18", lambda R: R["content"]["dof"], lambda v: str(int(v)), "num")
    add("5.3.2.1", "chi-square p", "< .001", lambda R: R["content"]["p"], fp, "p")
    add("5.3.2.1", "15 Hz: Null (minimum)", "0.5%", lambda R: R["content"]["pct_null_15"], pct1, "num")
    add("5.3.2.1", "semantic content (Semantic + Mixed) in all SLS trials", "just over 1%",
        lambda R: (1.0 < R["content"]["pct_semantic_any_sls"] < 1.5,
                   f"{R['content']['pct_semantic_any_sls']:.2f}%"), None, "claim")
    add("5.3.2.1", "labelled SLS trials (not printed; Figure 5.3A)", "(1,284)",
        lambda R: R["content"]["n"], fint, "info")
    add("5.3.2.1", "IRR AC1 0.901 / 0.821, kappa 0.735 / 0.577", "not recomputed",
        lambda R: "unaffected: P358 trial 8 was never classified or rated", str, "info")

    # ------------------------------------------------------------ S5.4 text and Table S5.2
    add("S5.4 text; Table S5.2 caption", "SLS trials in trial-order models", "1,281",
        lambda R: R["order_meta"]["n_exp"], fint, "num")
    add("S5.4 text; Table S5.2 caption", "validation trials in trial-order models", "296",
        lambda R: R["order_meta"]["n_val"], fint, "num")
    s52 = [
        ("Exp.", "detail", "−0.0193 (0.0046)", "17.55", "< .001", "−0.289", "0.79", "0.06", "1.000", "0.66"),
        ("Exp.", "r_directionality", "+0.0090 (0.0054)", "2.77", ".632", "+0.135", "0.17", "0.00", "1.000", "0.00"),
        ("Exp.", "theta_directionality", "−0.0152 (0.0055)", "7.71", ".044", "−0.228", "0.49", "0.00", "1.000", "0.00"),
        ("Exp.", "brightness", "−0.0085 (0.0050)", "2.87", ".632", "−0.127", "0.15", "0.02", "1.000", "0.29"),
        ("Exp.", "contrast", "−0.0161 (0.0046)", "12.04", ".005", "−0.242", "0.56", "1.13", ".717", "2.59"),
        ("Val.", "detail", "+0.0084 (0.0093)", "0.82", "1.000", "+0.126", "0.14", "2.16", ".637", "3.01"),
        ("Val.", "r_directionality", "−0.0019 (–)", "0.22", "1.000", "−0.029", "0.01", "0.00", "1.000", "0.00"),
        ("Val.", "theta_directionality", "+0.0027 (0.0046)", "0.40", "1.000", "+0.041", "0.02", "0.00", "1.000", "0.00"),
        ("Val.", "brightness", "−0.0117 (0.0073)", "2.56", ".877", "−0.175", "0.28", "0.04", "1.000", "0.17"),
        ("Val.", "contrast", "−0.0168 (0.0077)", "5.15", ".232", "−0.252", "0.59", "0.00", "1.000", "0.00"),
    ]
    for tr, f, bse, chi_t, p_t, drift, pct_t, chi_s, p_s, pct_s in s52:
        T = "Experimental" if tr == "Exp." else "Validation"
        loc = f"Table S5.2, {tr} {S52_LABEL[f]}"
        add(loc, "b per trial (SE)", bse, lambda R, T=T, f=f: (_t(R, T, f, "b"), _t(R, T, f, "se")), _bse, "num")
        add(loc, "chi2(1) trial", chi_t, lambda R, T=T, f=f: _t(R, T, f, "chi2_trial"), f2, "num")
        add(loc, "p trial (Holm, m = 10)", p_t, lambda R, T=T, f=f: _t(R, T, f, "p_trial_holm"), _p3, "p")
        add(loc, "drift across experiment (SD)", drift, lambda R, T=T, f=f: _t(R, T, f, "drift"), lambda v: f"{v:+.3f}", "num")
        add(loc, "trial order (% variance)", pct_t, lambda R, T=T, f=f: _t(R, T, f, "pct_trial"), f2, "num")
        add(loc, "session chi2(1)", chi_s, lambda R, T=T, f=f: _t(R, T, f, "chi2_session"), f2, "num")
        add(loc, "session p (Holm, m = 10)", p_s, lambda R, T=T, f=f: _t(R, T, f, "p_session_holm"), _p3, "p")
        add(loc, "session (% variance)", pct_s, lambda R, T=T, f=f: _t(R, T, f, "pct_session"), f2, "num")
    add("S5.4 text", "detail drift (2 dp)", "−0.29 SD", lambda R: _t(R, "Experimental", "detail", "drift"), lambda v: f"{v:.2f} SD", "num")
    add("S5.4 text", "tangential drift", "−0.23 SD", lambda R: _t(R, "Experimental", "theta_directionality", "drift"), lambda v: f"{v:.2f} SD", "num")
    add("S5.4 text", "contrast drift", "−0.24 SD", lambda R: _t(R, "Experimental", "contrast", "drift"), lambda v: f"{v:.2f} SD", "num")
    add("S5.4 text", "radial and brightness trial p ('both p = .632')", "both .632",
        lambda R: _both(_p3(_t(R, 'Experimental', 'r_directionality', 'p_trial_holm')),
                        _p3(_t(R, 'Experimental', 'brightness', 'p_trial_holm'))), str, "num")
    add("S5.4 text", "radial and brightness drift ('< 0.14 SD')", "< 0.14 SD",
        lambda R: (bool(max(abs(_t(R, 'Experimental', 'r_directionality', 'drift')),
                            abs(_t(R, 'Experimental', 'brightness', 'drift'))) < 0.14),
                   f"max |drift| = {max(abs(_t(R, 'Experimental', 'r_directionality', 'drift')), abs(_t(R, 'Experimental', 'brightness', 'drift'))):.3f}"),
        None, "claim")
    add("S5.4 text; 5.4.2", "significant trial-position effects in Exp. data (Holm < .05)",
        "3 (detail level, tangential, contrast)",
        lambda R: _sig_list(R), str, "num")
    add("S5.4 text", "validation contrast b", "−0.0168", lambda R: _t(R, "Validation", "contrast", "b"), f4, "num")
    add("S5.4 text", "validation contrast drift", "−0.25 SD", lambda R: _t(R, "Validation", "contrast", "drift"), lambda v: f"{v:.2f} SD", "num")
    add("S5.4 text", "validation: all trial p > .23", "all p > .23",
        lambda R: (bool((R["order"].query("Trials == 'Validation'")["p_trial_holm"] > 0.23).all()),
                   f"min = {_val_rows(R)['p_trial_holm'].min():.3f}"), None, "claim")
    add("S5.4 text", "validation: drift < 0.26 SD", "< 0.26 SD",
        lambda R: (bool((R["order"].query("Trials == 'Validation'")["drift"].abs() < 0.26).all()),
                   f"max |drift| = {_val_rows(R)['drift'].abs().max():.3f}"), None, "claim")
    add("S5.4 text", "session intercept significant for no feature", "none significant",
        lambda R: (bool((R["order"]["p_session_holm"] >= 0.05).all()),
                   f"min Holm p = {R['order']['p_session_holm'].min():.3f}"), None, "claim")
    add("S5.4 text; 5.4.2", "trial position % variance ('under 0.8%'; '< 1%')", "under 0.8%",
        lambda R: (bool(R["order"]["pct_trial"].max() < 0.8), f"max {R['order']['pct_trial'].max():.2f}%"),
        None, "claim")
    add("S5.4 text", "session % variance ('at most 3.0%', 1 dp)", "at most 3.0%",
        lambda R: (bool(round(R["order"]["pct_session"].max(), 1) <= 3.0),
                   f"max {R['order']['pct_session'].max():.2f}%"), None, "claim")
    add("S5.4 text", "max change in frequency coefficients", "0.021 SD",
        lambda R: R["order_meta"]["max_shift"], lambda v: f"{v:.3f} SD", "num")
    add("S5.4 text", "min correlation of frequency coefficients", "r ≥ 0.999",
        lambda R: (bool(R["order_meta"]["min_r"] >= 0.999), f"r = {R['order_meta']['min_r']:.4f}"),
        None, "claim")
    return L


def _means_txt(R, feat):
    m = _m(R, feat)
    return ", ".join(f"{(str(int(h)) if float(h).is_integer() else str(h))}: {v:.3f}" for h, v in m.items())


def _g(v):
    t = f"{v:.2f}"
    return "0" if float(t) == 0 else t


def _both(a, b):
    return f"both {a}" if a == b else f"{a} and {b}"


def _val_rows(R):
    return R["order"][R["order"]["Trials"] == "Validation"]


def _beyond15(R):
    ok = True
    for f, sign in (("brightness", +1), ("detail", -1), ("r_directionality", -1), ("contrast", -1)):
        m = _m(R, f).loc[[15.0, 20.0, 40.0, 80.0]].to_numpy()
        d = (m[1:] - m[:-1]) * sign
        ok &= bool((d > 0).all())
    return ok


def _sig_list(R):
    o = R["order"]
    s = o[(o["Trials"] == "Experimental") & (o["p_trial_holm"] < 0.05)]["feature"].tolist()
    return f"{len(s)} ({', '.join(S52_LABEL[x] for x in s)})"


def _bse(v):
    b, se = v
    import numpy as np
    return f"{b:+.4f} (-)" if not np.isfinite(se) else f"{b:+.4f} ({se:.4f})"


def _p3(p):
    return "< .001" if p < 0.001 else f"{p:.3f}".replace("0.", ".", 1)
