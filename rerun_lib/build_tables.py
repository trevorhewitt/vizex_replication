"""Step 1: rebuild the analysis tables from the committed CSVs.

Without the raw images, 010_build_index and 015_feature_extraction cannot be
executed. This module reproduces exactly the two parts of them that depend on
the analysis set:

* 010_build_index, cell "apply exclusions and write the master trial table":
  exclusion_reason is '' / 'not_a_trial' / 'no_cropped_image' / 'manual', and
  in_analysis = (exclusion_reason == ''). Manual exclusions come from
  015_tables/manual_exclusions.csv.
* 015_feature_extraction, add_feature(): the edge features are rescaled so that
  the 0.95 quantile of the raw values over the IN-ANALYSIS TRIALS maps to 1.0
  (float32, same casting as the notebook); the same scale is applied to the
  target images. Brightness and contrast are not rescaled.

The raw feature values (`<feature>_raw`) are taken from the committed
vizex_features.csv / vizex_targets_features.csv, which 015 computed from the
images. Unchanged metadata cells are written back byte-for-byte.
"""

import numpy as np
import pandas as pd

from .common import (FEATURES, MANUAL_EXCLUSIONS_CSV, TARGETS_FEATURES_CSV,
                     TRIALS_FEATURES_CSV, TRIALS_MASTER_CSV)

NON_TRIAL_CODES = {"9", "POSTQ"}          # as in 010_build_index
ANALYSIS_FILE_TYPE = "cropped"
NORM_QUANTILE = 0.95                      # as in 015_feature_extraction
NORMALISE = {"detail": True, "r_directionality": True, "theta_directionality": True,
             "brightness": False, "contrast": False}


def read_str(path) -> pd.DataFrame:
    """Read every cell as text so unchanged cells are written back identically."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def apply_exclusions(trials: pd.DataFrame, manual: pd.DataFrame | None) -> pd.DataFrame:
    t = trials.copy()
    reason = pd.Series("", index=t.index)
    reason[t["condition_code"].isin(NON_TRIAL_CODES)] = "not_a_trial"
    no_img = (t["png_filename"] == "") & (reason == "")
    reason[no_img] = f"no_{ANALYSIS_FILE_TYPE}_image"
    if manual is not None and len(manual):
        keys = set(zip(manual["participant_code"].astype(str),
                       manual["trialn"].astype(int)))
        hit = [(p, int(n)) in keys and r == ""
               for p, n, r in zip(t["participant_code"], t["trialn"], reason)]
        reason[hit] = "manual"
    t["exclusion_reason"] = reason
    t["in_analysis"] = np.where(reason == "", "True", "False")
    return t


def check_manual_rows(trials: pd.DataFrame, manual: pd.DataFrame) -> list[str]:
    """Each manual exclusion must name exactly one trial (and its file, if given)."""
    problems = []
    for _, m in manual.iterrows():
        sel = trials[(trials["participant_code"] == str(m["participant_code"]))
                     & (trials["trialn"].astype(int) == int(m["trialn"]))]
        if len(sel) != 1:
            problems.append(f"{m['participant_code']} trial {m['trialn']}: matches {len(sel)} rows")
        elif "png_filename" in m and str(m.get("png_filename", "")).strip():
            if sel["png_filename"].iloc[0] != str(m["png_filename"]).strip():
                problems.append(f"{m['participant_code']} trial {m['trialn']}: filename mismatch")
    return problems


def _normalise(raw_text: pd.Series, scale: float) -> np.ndarray:
    raw32 = pd.to_numeric(raw_text, errors="coerce").to_numpy(np.float64).astype(np.float32)
    return raw32 * scale  # float32 * python float -> float32, as in 015


def renormalise(trials_master: pd.DataFrame, features: pd.DataFrame,
                targets: pd.DataFrame):
    """Rebuild vizex_features / vizex_targets_features for the in-analysis set.

    Returns (features_out, targets_out, scales, n_missing)."""
    master = trials_master[trials_master["in_analysis"] == "True"].reset_index(drop=True)
    raw_cols = [f"{f}_raw" for f in FEATURES]
    src = features.set_index("png_filename")[raw_cols]
    missing = [p for p in master["png_filename"] if p not in src.index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} in-analysis trial(s) have no raw feature values in "
            f"vizex_features.csv (first: {missing[:2]}). Re-run 015_feature_extraction "
            f"with the images in 005_cleaned_data/ (python rerun.py --with-images).")
    out = master.copy()
    raw = src.loc[out["png_filename"]].reset_index(drop=True)

    tg = targets.copy()
    scales = {}
    for f in FEATURES:
        raw32 = pd.to_numeric(raw[f"{f}_raw"]).to_numpy(np.float64).astype(np.float32)
        if NORMALISE[f]:
            q = float(np.nanquantile(raw32[np.isfinite(raw32)], NORM_QUANTILE))
            scale = 1.0 / q if np.isfinite(q) and q > 0 else 1.0
        else:
            scale = 1.0
        scales[f] = scale
        out[f"{f}_raw"] = raw[f"{f}_raw"].to_numpy()
        out[f] = _normalise(raw[f"{f}_raw"], scale)
        tg[f] = _normalise(tg[f"{f}_raw"], scale)
    # keep the committed column order
    out = out[list(features.columns)]
    tg = tg[list(targets.columns)]
    return out, tg, scales


def _same_up_to_float32(a_text: str, b_text: str, rtol: float = 1e-6) -> bool:
    """Same rows, same text in every non-feature cell, feature values within rtol."""
    import io
    a = pd.read_csv(io.StringIO(a_text), dtype=str, keep_default_na=False)
    b = pd.read_csv(io.StringIO(b_text), dtype=str, keep_default_na=False)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    num = [c for c in a.columns if c in FEATURES]
    if not a.drop(columns=num).equals(b.drop(columns=num)):
        return False
    x = a[num].apply(pd.to_numeric).to_numpy(float)
    y = b[num].apply(pd.to_numeric).to_numpy(float)
    return bool(np.allclose(x, y, rtol=rtol, atol=1e-12, equal_nan=True))


def to_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Typed copy for analysis (the string copies are only for writing)."""
    d = df.copy()
    for c in d.columns:
        if c in ("condition_hz", "trialn", "n_files") or c in FEATURES or c.endswith("_raw"):
            d[c] = pd.to_numeric(d[c].replace("", np.nan), errors="coerce")
    if "in_analysis" in d.columns:
        d["in_analysis"] = d["in_analysis"].astype(str).eq("True")
    return d


def build(checks, write: bool = True, apply_manual: bool = True,
          trials_master: pd.DataFrame | None = None,
          features: pd.DataFrame | None = None,
          targets: pd.DataFrame | None = None, label: str = "current"):
    """Apply exclusions, rebuild the feature tables, optionally write them back.

    Returns dict with typed trials_master / features / targets and the scales."""
    tm = read_str(TRIALS_MASTER_CSV) if trials_master is None else trials_master
    ft = read_str(TRIALS_FEATURES_CSV) if features is None else features
    tg = read_str(TARGETS_FEATURES_CSV) if targets is None else targets

    manual = None
    if apply_manual and MANUAL_EXCLUSIONS_CSV.exists():
        manual = pd.read_csv(MANUAL_EXCLUSIONS_CSV, dtype=str, keep_default_na=False)
        problems = check_manual_rows(tm, manual)
        checks.hard(f"[{label}] manual_exclusions.csv rows each match one trial and file",
                    len(problems), 0, "; ".join(problems))

    tm_new = apply_exclusions(tm, manual)
    tg = tg[tg["exists"] == "True"].reset_index(drop=True) if "exists" in tg else tg
    ft_new, tg_new, scales = renormalise(tm_new, ft, tg)

    if write:
        out = []
        for df, path in ((tm_new, TRIALS_MASTER_CSV), (ft_new, TRIALS_FEATURES_CSV),
                         (tg_new, TARGETS_FEATURES_CSV)):
            before = path.read_text(encoding="utf-8") if path.exists() else ""
            text = df.to_csv(index=False, lineterminator="\n")
            if text == before:
                checks.info(f"[{label}] {path.name}", "unchanged")
            elif before and _same_up_to_float32(before, text):
                # numpy 2 can differ from numpy 1.x (used for the committed files) in the
                # last float32 digit of the rescaled features; keep the committed file
                df = read_str(path)
                checks.info(f"[{label}] {path.name}", "unchanged (differences only in the last float32 digit)")
            else:
                path.write_text(text, encoding="utf-8")
                checks.info(f"[{label}] {path.name}", "UPDATED")
            out.append(df)
        tm_new, ft_new, tg_new = out

    return {
        "trials_master": to_numeric_features(tm_new),
        "features": to_numeric_features(ft_new),
        "targets": to_numeric_features(tg_new),
        "scales": scales,
        "manual": manual,
    }
