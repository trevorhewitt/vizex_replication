# Rerun of Chapter 5 with 1,580 trials

This branch (`rerun/trial-counts-1580`) excludes P358 trial 8 (a spurious
2.5 Hz re-save; the scheduled trial was lost and not recovered) and regenerates
every Chapter 5 table and figure from the committed CSVs. It needs no raw images
and no API key.

## Setup

Tested with Python 3.11 in two environments, with identical results:
(a) numpy 1.26.4, pandas 2.2.3, scipy 1.13.1, statsmodels 0.14.4,
matplotlib 3.9.2; (b) numpy 2.4, pandas 3.0, scipy 1.17, statsmodels 0.15.
Any Python 3.10-3.12 should work.

```bash
git fetch origin
git checkout rerun/trial-counts-1580
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run (one command, from the repo root)

```bash
python rerun.py
```

Runtime: about 2-4 minutes on a laptop. Options:

* `--with-images`: also executes `011_notebooks_v2/010_build_index` and
  `015_feature_extraction` on `005_cleaned_data/`. This is the only way to
  regenerate the Figure S5.4 heatmaps, which are computed from the pixels. It
  also checks that the image-derived features equal the CSV-derived ones.
  Expect tens of minutes.
* `--notebooks`: also executes notebooks 030/035/036 in place, which updates
  their outputs and `015_tables/table_*`, and compares them with the scripts.
* `--no-baseline`: skips the reproduction on the pre-exclusion data, which
  needs `git`.

Exit code 0 means every hard check passed. A failed hard check stops the run
with a non-zero exit code.

## What `rerun.py` does

1. **Step 1: analysis set.** Applies `015_tables/manual_exclusions.csv` exactly
   as `010_build_index` does. It then rebuilds `015_tables/vizex_features.csv`
   and `vizex_targets_features.csv` as `015_feature_extraction` would: the edge
   features are re-normalised to the 0.95 quantile of the in-analysis trials.
   Step 1 is idempotent and reports `unchanged` on this branch.
2. **Step 2: analyses.** Runs the validation models (Table S5.3, Figure 5.1),
   the experimental omnibus models (Table S5.4, Figure 5.3B-F), the planned
   contrasts including the 10 vs 15 Hz follow-up, and the condition means. It
   also runs trial order / session (Table S5.2; the logic of 036 at 6e52ac6)
   and the content categories from the stored GPT-4o labels (Figure 5.3A,
   χ²(18)). No API call is made.
3. **Step 3: baseline.** Runs the same code on the pre-exclusion tables at
   commit 479ccfb. This reproduces the thesis values.
4. **Step 4: part D.** Runs an independent random-intercept LME written from
   scratch in `rerun_lib/verify.py`, plus hand checks of the Wald χ², ICC,
   CV%, Holm, drift and the chi-square (with a Monte Carlo p).
5. **Step 5: report.** Writes `OLD_VS_NEW.md`, comparing every thesis number
   with the new value.

## Checks printed as PASS / FAIL (hard) or PASS / WARN (soft)

Hard checks:

* `trials_master.csv`: 1,581 rows; 1,580 `in_analysis`; 1 excluded, and it is
  P358 trial 8 (`manual`).
* `vizex_features.csv` has exactly the 1,580 in-analysis trials: 1,186
  experimental, 98 control and 296 validation, from 99 participants.
* SLS trials: 1,284. Per frequency (2.5/5/10/15/20/40/80 Hz):
  197/198/198/197/198/198/98.
* P358 trial 9 (blank 40 Hz) is kept. The December trials (P306 101 and 102,
  P324 101) are kept in the main analyses.
* Model sizes:
  * validation models: n = 296;
  * Table S5.4 models: n = 1,284, df = 6;
  * trial-order models: 1,281 SLS (1,183 experimental + 98 control) and 296
    validation, with the 3 December trials excluded, span 15, and the P368
    make-up trial at position 4.5;
  * 10 vs 15 Hz follow-up: n = 395.
* The VLM labels cover exactly the 1,284 in-analysis SLS trials.

Soft checks:

* The 10 vs 15 Hz follow-up (b, SE, z) is unchanged at the printed precision.
* The validation slopes, SEs, z and p are unchanged.
* The independent refits agree with statsmodels.

## Send back

The whole `rerun_outputs/` folder (about 2.5 MB):

* `OLD_VS_NEW.md` / `.csv`: thesis value vs pre-exclusion reproduction vs new.
* `CHECKS.csv`: every PASS/FAIL/WARN.
* `VERIFICATION.csv`: part D.
* `RUN_INFO.md`: library versions.
* `tables/`: Tables S5.2, S5.3 and S5.4 in thesis layout plus `_raw`
  full-precision versions, the contrasts, condition means, content counts and
  χ², and the normalisation scales.
* `figures/`: Figure 5.1 A-E, Figure 5.3 A-F plus keys, and (with
  `--with-images`) `S5.4_heatmaps/`.

## Notes

* **Optimiser choice.** With statsmodels 0.14.x, L-BFGS can report
  "converged" on a degenerate fit with log-likelihood = +inf and participant
  variance 0. The notebooks' rule ("first optimiser that converges") would then
  give wrong Table S5.4 values; statsmodels 0.15 raises instead.
  `rerun_lib/common.fit_mixedlm` and the notebooks now reject non-finite fits.
  On this data every remaining optimiser reaches the same maximum, which is the
  one reported in the thesis.
* **Stored VLM labels.** They come from `vizex_qual` (branch `laptop`); see
  `015_tables/vlm_content_labels_SOURCE.md`.
