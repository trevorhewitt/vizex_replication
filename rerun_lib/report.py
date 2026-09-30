"""OLD_VS_NEW.md: thesis value vs pre-exclusion reproduction vs new value."""

import re

import numpy as np
import pandas as pd


def _norm(s) -> str:
    s = str(s).replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", "", s).lower()


def _p_from_text(s):
    s = _norm(s)
    if s.startswith("<"):
        return float(s[1:]) / 2
    try:
        return float(s)
    except ValueError:
        return np.nan


def _evaluate(getter, fmt, kind, R):
    if R is None:
        return None, "n/a"
    try:
        v = getter(R)
    except Exception as e:  # noqa: BLE001 - report, don't crash the report
        return None, f"ERROR: {e}"
    if kind == "claim":
        ok, ev = v
        return ok, ("holds" if ok else "DOES NOT HOLD") + (f" ({ev})" if ev else "")
    return v, fmt(v) if fmt else str(v)


def build_rows(items, R_new, R_base):
    rows = []
    for i, (loc, qty, thesis, getter, fmt, kind) in enumerate(items, 1):
        v_new, t_new = _evaluate(getter, fmt, kind, R_new)
        v_base, t_base = _evaluate(getter, fmt, kind, R_base)
        if kind == "claim":
            if v_new:
                status = "holds"
            elif R_base is not None and v_base:
                status = "CLAIM NO LONGER HOLDS (changed by the exclusion)"
            elif R_base is not None:
                status = "claim does not hold (pre-existing: same before the exclusion)"
            else:
                status = "CLAIM DOES NOT HOLD"
        elif kind == "info":
            status = "info"
        else:
            same_new = _norm(t_new) == _norm(thesis)
            status = "unchanged" if same_new else "CHANGED"
            if kind == "p" and v_new is not None:
                th = _p_from_text(thesis)
                if np.isfinite(th) and (th < 0.05) != (float(v_new) < 0.05):
                    status = "SIGNIFICANCE VERDICT CHANGED"
            if R_base is not None and _norm(t_base) != _norm(thesis) and not t_base.startswith("ERROR"):
                status += "; thesis differs from pre-exclusion reproduction"
        rows.append({"#": i, "Location": loc, "Quantity": qty, "Thesis (old)": thesis,
                     "Pre-exclusion reproduction": t_base, "New": t_new, "Status": status})
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(esc(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write(rows: pd.DataFrame, path, title: str, preamble: str) -> None:
    n = len(rows)
    st = rows["Status"]
    changed = rows[st.str.startswith("CHANGED") | st.str.startswith("SIGNIFICANCE")
                   | st.str.startswith("CLAIM")]
    verdict = rows[st.str.startswith("SIGNIFICANCE") | st.str.startswith("CLAIM NO LONGER")]
    pre = rows[st.str.contains("pre-existing") | st.str.contains("thesis differs")]
    summary = [
        f"# {title}", "", preamble, "",
        "## Summary", "",
        f"* Items compared: {n}",
        f"* Unchanged at the precision printed in the thesis: {int((st == 'unchanged').sum())}",
        f"* Changed (thesis text/table needs an edit): {len(changed)}",
        f"* **Significance verdicts or claims changed by the exclusion: {len(verdict)}**",
        f"* Pre-existing discrepancies (thesis value not reproduced even before the exclusion, "
        f"or claim already false): {len(pre)}",
        "",
        "Status key: `unchanged` = new value prints exactly as in the thesis; `CHANGED` = "
        "different at the printed precision, same significance verdict; `SIGNIFICANCE VERDICT "
        "CHANGED` = crosses p = .05; `CLAIM NO LONGER HOLDS` = a verbal claim that was true "
        "before the exclusion is now false; `thesis differs from pre-exclusion reproduction` = "
        "the thesis value was not what the code produced on the old data either (a pre-existing "
        "reporting issue, not caused by the exclusion).", "",
    ]
    body = []
    if len(verdict):
        body += ["## Verdict or claim changes caused by the exclusion", "", _md_table(verdict), ""]
    if len(changed):
        body += ["## All items that differ from the thesis", "", _md_table(changed), ""]
    if len(pre):
        body += ["## Pre-existing discrepancies", "", _md_table(pre), ""]
    body += ["## Full comparison", "", _md_table(rows), ""]
    path.write_text("\n".join(summary + body), encoding="utf-8")
    rows.to_csv(path.with_suffix(".csv"), index=False)
