# Provenance of `vlm_content_labels.csv`

These are the stored GPT-4o content classifications used for thesis Section
5.2.2.5 / 5.3.2.1 and Figure 5.3A. They were **not** regenerated: no API call
was made. The CSV is a lossless extract of the fields needed for the analysis.

* Source repository: `trevorhewitt/vizex_qual`, branch `laptop`,
  commit `f64ace90d5c6b756e0d0310068efea83c6affe3f`.
* Source files: `pipeline_results/<png stem>_openai.json`, 1,284 files, one per
  SLS trial, written by `ai_qual.ipynb` (model `gpt-4o-2024-08-06`,
  temperature 0; committed 2026-05-01/02 in c1d7d7e and 35323c2 and unchanged
  since).
* SHA-256 of the 1,284 JSON files concatenated in sorted path order:
  `32f3fa984330dc19c00233911a9b6affd2d257bcc9c197c435a9b93398d41d19`.
* Columns copied: `category`, `geometric_present`, `semantic_present`,
  `confidence_score`. `participant_code`, `trialn` and `condition_code` are
  parsed from the file name. The free-text `reasoning` field is not copied.
* `category` agrees with the two booleans for all 1,284 rows.

The 1,284 labelled trials are exactly the in-analysis SLS trials
(1,186 experimental + 98 control). P358 trial 8 (the excluded spurious 2.5 Hz
re-save) was never classified, so the stored chi-square and percentages were
already computed without it; `rerun.py` asserts this.
