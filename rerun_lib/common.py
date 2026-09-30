"""Shared paths, constants, model helpers and the PASS/FAIL check registry."""

import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SEED = 42

PROJECT_PATH = Path(__file__).resolve().parent.parent
TABLES_PATH = PROJECT_PATH / "015_tables"
OUT_PATH = PROJECT_PATH / "rerun_outputs"
FIG_PATH = OUT_PATH / "figures"
TAB_PATH = OUT_PATH / "tables"

TRIALS_MASTER_CSV = TABLES_PATH / "trials_master.csv"
TARGETS_MASTER_CSV = TABLES_PATH / "targets_master.csv"
MANUAL_EXCLUSIONS_CSV = TABLES_PATH / "manual_exclusions.csv"
TRIALS_FEATURES_CSV = TABLES_PATH / "vizex_features.csv"
TARGETS_FEATURES_CSV = TABLES_PATH / "vizex_targets_features.csv"
VLM_LABELS_CSV = TABLES_PATH / "vlm_content_labels.csv"

# Column name -> label used in the thesis tables and figures
FEATURE_LABELS = {
    "detail": "detail level",
    "r_directionality": "radial orientation",
    "theta_directionality": "tangential orientation",
    "brightness": "brightness",
    "contrast": "contrast",
}
FEATURES = list(FEATURE_LABELS)

# Strobe condition codes -> Hz (as in 010_build_index / 015_feature_extraction)
HZ_MAP = {1: 2.5, 2: 5, 3: 10, 4: 15, 5: 20, 6: 40, 7: 80}
CONTROL_CODE = "7"

# The analysis set the thesis reports (Section 5.2.1.4, Table S5.4, Section S5.4)
EXPECTED = {
    "participants": 99,
    "in_analysis": 1580,
    "experimental": 1186,
    "control": 98,
    "validation": 296,
    "sls": 1284,
    "sls_trial_order": 1281,
    "experimental_trial_order": 1183,
    "validation_trial_order": 296,
    "per_hz": {2.5: 197, 5.0: 198, 10.0: 198, 15.0: 197, 20.0: 198, 40.0: 198, 80.0: 98},
    "trials_master_rows": 1581,
    "excluded": 1,
}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def quiet_warnings() -> None:
    # statsmodels emits boundary/singular-covariance warnings for the validation
    # models (participant variance ~0); they are expected and reported in the tables.
    warnings.filterwarnings("ignore")


def ensure_dirs() -> None:
    for p in (OUT_PATH, FIG_PATH, TAB_PATH):
        p.mkdir(parents=True, exist_ok=True)


OPTIMISERS = ("lbfgs", "bfgs", "cg", "powell", "nm")   # order used in 030/035
LLF_TOL = 1e-4


def fit_mixedlm(formula: str, data: pd.DataFrame, reml: bool = False):
    """Random participant intercept (ML by default), robust to optimiser failure.

    030/035 take the first optimiser, in the order above, that reports
    convergence. With statsmodels 0.14.x, L-BFGS can report convergence at a
    degenerate singular fit (log-likelihood = +inf, participant variance 0);
    with 0.15 the same call raises instead. To make the result independent of
    the installed version, every optimiser is tried, fits with a non-finite
    log-likelihood are rejected, and the first optimiser (in the notebook's
    order) whose log-likelihood is within LLF_TOL of the best is used. On the
    committed data this reproduces the thesis tables exactly.
    Returns (model, result, method)."""
    md = smf.mixedlm(formula, data, groups=data["participant_code"])
    fits = []
    for method in OPTIMISERS:
        try:
            r = md.fit(reml=reml, method=method, maxiter=2000, disp=False)
        except Exception:
            continue
        if getattr(r, "converged", False) and np.isfinite(r.llf):
            fits.append((method, r))
    if not fits:
        raise RuntimeError(f"no optimiser gave a finite, converged fit for '{formula}'")
    best = max(r.llf for _, r in fits)
    for method, r in fits:
        if r.llf >= best - LLF_TOL:
            return md, r, method
    raise AssertionError("unreachable")


def format_p(p, small: float = 0.001, decimals: int = 3) -> str:
    """'< .001' for very small p, otherwise '.004' style with no leading zero."""
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return ""
    if p < small:
        return "< " + f"{small:.{decimals}f}".lstrip("0")
    text = f"{p:.{decimals}f}"
    return text.lstrip("0") if text.startswith("0") else text


class Checks:
    """Collects PASS/FAIL lines. Hard checks abort the run after the stage that
    produced them; soft checks are reported (and flagged in OLD_VS_NEW.md)."""

    def __init__(self):
        self.rows = []

    def _add(self, kind, name, ok, observed, expected, note=""):
        status = "PASS" if ok else ("FAIL" if kind == "hard" else "WARN")
        self.rows.append({"kind": kind, "check": name, "status": status,
                          "observed": observed, "expected": expected, "note": note})
        line = f"[{status}] {name}: observed {observed}; expected {expected}"
        if note:
            line += f"  ({note})"
        print(line, flush=True)
        return ok

    def hard(self, name, observed, expected, note=""):
        return self._add("hard", name, observed == expected, observed, expected, note)

    def soft(self, name, ok, observed, expected, note=""):
        return self._add("soft", name, bool(ok), observed, expected, note)

    def info(self, name, observed, note=""):
        self.rows.append({"kind": "info", "check": name, "status": "INFO",
                          "observed": observed, "expected": "", "note": note})
        print(f"[INFO] {name}: {observed}" + (f"  ({note})" if note else ""), flush=True)

    def abort_if_failed(self, stage: str) -> None:
        failed = [r for r in self.rows if r["kind"] == "hard" and r["status"] == "FAIL"]
        if failed:
            self.write()
            print(f"\nABORTING after '{stage}': {len(failed)} hard check(s) failed. "
                  f"See rerun_outputs/CHECKS.csv.", file=sys.stderr, flush=True)
            sys.exit(1)

    def write(self) -> None:
        ensure_dirs()
        pd.DataFrame(self.rows).to_csv(OUT_PATH / "CHECKS.csv", index=False)

    def summary(self) -> str:
        df = pd.DataFrame(self.rows)
        if df.empty:
            return "no checks"
        counts = df["status"].value_counts().to_dict()
        return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
