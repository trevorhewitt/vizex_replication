# Thesis edit list: 1,580-trial correction (Chapters 5 and 6.1)

Source: `HEWITT_THESIS_FULL_9.docx`. Every replacement value below was
recomputed by the committed reruns:

* `vizex_replication` and `thesis_meta_analysis`, branch
  `rerun/trial-counts-1580`;
* files `rerun_outputs/OLD_VS_NEW.md` in each repo.

Both reruns were run twice, with statsmodels 0.14.4 / pandas 2.2 and with
statsmodels 0.15 / pandas 3.0, and gave identical results. On the old data the
same code reproduces every number currently in Tables S5.2, S5.3, S5.4 and 6.4
exactly (apart from the pre-existing issues noted below). No number is changed
here unless it was recomputed.

**No significance verdict and no conclusion changes.** Every test that was
significant is still significant, at the same threshold. The only wording
change forced by the new numbers is edit 1 ("> 113" becomes "> 112").

Search tips:

* The docx may use the χ symbol and a superscript 2 where this list writes
  `chi^2`, so search for the numeric part instead.
* Table cells hold one number each: search for the unique number shown, then
  edit the row.
* `−` is the Unicode minus used in Section S5.4 and Table S5.2. Table 6.4 uses
  an ASCII hyphen `-`.

---

## Chapter 5

### Edit 1: Section 5.3.2.2, first sentence (wording; verdict unchanged)

* **Ctrl-F:** `> 113`
* **Current:** `(Wald chi^2(6) > 113, p < .001 for all five models; see Supplementary Table S5.4 for full results)`
* **Replace with:** `(Wald chi^2(6) > 112, p < .001 for all five models; see Supplementary Table S5.4 for full results)`
* **Why:** the smallest χ² (brightness) is now 112.1, previously 113.4. All
  five models remain at p < .001.

### Edit 2: Supplementary Section S5.4, paragraph 3

* **Ctrl-F:** `both p = .632`
* **Current:** `(both p = .632; drift across experiment < 0.14 SD)`
* **Replace with:** `(both p = .615; drift across experiment < 0.14 SD)`
* Still not significant, so the verdict is unchanged. The other numbers in
  this paragraph and the next are unchanged: −0.0193, −0.29, −0.23, −0.24,
  −0.0168, −0.25, p > .23, < 0.26 SD, under 0.8%, at most 3.0%, 0.021 SD and
  r ≥ 0.999.

### Edit 3: Table S5.2, the five "Exp." rows

The "Val." rows are unchanged. Columns: b per trial (SE) | χ²(1) | p |
Drift | Trial order % | Session χ²(1) | Session p | Session %.

| Row | Ctrl-F | Current | Replace with |
|---|---|---|---|
| Exp. detail level | `17.55` | `−0.0193 (0.0046) \| 17.55 \| < .001 \| −0.289 \| 0.79 \| 0.06 \| 1.000 \| 0.66` | `−0.0193 (0.0046) \| 17.54 \| < .001 \| −0.289 \| 0.79 \| 0.06 \| 1.000 \| 0.67` |
| Exp. radial | `2.77` | `+0.0090 (0.0054) \| 2.77 \| .632 \| +0.135 \| 0.17 \| 0.00 \| 1.000 \| 0.00` | `+0.0090 (0.0054) \| 2.76 \| .615 \| +0.135 \| 0.17 \| 0.00 \| 1.000 \| 0.00` |
| Exp. tangential | `7.71` | `−0.0152 (0.0055) \| 7.71 \| .044 \| −0.228 \| 0.49 \| 0.00 \| 1.000 \| 0.00` | `−0.0152 (0.0055) \| 7.70 \| .044 \| −0.228 \| 0.49 \| 0.00 \| 1.000 \| 0.00` |
| Exp. brightness | `2.87` | `−0.0085 (0.0050) \| 2.87 \| .632 \| −0.127 \| 0.15 \| 0.02 \| 1.000 \| 0.29` | `−0.0085 (0.0050) \| 2.91 \| .615 \| −0.128 \| 0.16 \| 0.00 \| 1.000 \| 0.13` |
| Exp. contrast | `12.04` | `−0.0161 (0.0046) \| 12.04 \| .005 \| −0.242 \| 0.56 \| 1.13 \| .717 \| 2.59` | `−0.0161 (0.0046) \| 12.05 \| .005 \| −0.242 \| 0.56 \| 1.19 \| .686 \| 2.66` |

The same three features remain significant (detail, tangential, contrast).
The caption is unchanged; "1,281 SLS and 296 validation trials" is now
exactly what was analysed.

### Edit 4: Table S5.3 (validation), three cells

| Row, column | Ctrl-F / current | Replace with |
|---|---|---|
| detail level, random-intercept variance | `0.019314` | `0.019307` |
| detail level, residual variance | `0.064122` | `0.064096` |
| tangential orientation, residual variance | `0.008976` | `0.008974` |

All slopes, SEs, z and p values are identical. The change comes only from
recomputing the P95 normalisation constant over 1,580 rather than 1,581
trials: detail scale ×0.99980, tangential ×0.99988. This is what
`015_feature_extraction` does, and it rescales the stimulus and recreation
values of the validation data alike.

### Edit 5: Table S5.4 (experimental), all five rows

The n column already reads 1284. The other numbers in the table were computed
from 1,285 trials; they are now from 1,284.

| Row | Ctrl-F | Current | Replace with |
|---|---|---|---|
| detail level | `130.2` | `1284 \| 0.16 \| 130.2 \| 6 \| < .001 \| 0.19 \| 116.28% \| 0.406 \| 0.023849 \| 0.034949` | `1284 \| 0.16 \| 129.7 \| 6 \| < .001 \| 0.19 \| 116.30% \| 0.405 \| 0.023843 \| 0.034963` |
| radial orientation | `126.5` | `1284 \| 0.40 \| 126.5 \| 6 \| < .001 \| 0.24 \| 59.37% \| 0.158 \| 0.010817 \| 0.057817` | `1284 \| 0.41 \| 126.4 \| 6 \| < .001 \| 0.24 \| 59.38% \| 0.158 \| 0.010819 \| 0.057854` |
| tangential orientation | `117.6` | `1284 \| 0.44 \| 117.6 \| 6 \| < .001 \| 0.24 \| 55.56% \| 0.137 \| 0.009349 \| 0.059029` | `1284 \| 0.44 \| 117.2 \| 6 \| < .001 \| 0.24 \| 55.59% \| 0.137 \| 0.009356 \| 0.059057` |
| brightness | `113.4` | `1284 \| 0.78 \| 113.4 \| 6 \| < .001 \| 0.15 \| 19.61% \| 0.294 \| 0.009658 \| 0.023215` | `1284 \| 0.78 \| 112.1 \| 6 \| < .001 \| 0.15 \| 19.52% \| 0.294 \| 0.009584 \| 0.023022` |
| contrast | `125.1` | `1284 \| 0.14 \| 125.1 \| 6 \| < .001 \| 0.09 \| 61.90% \| 0.390 \| 0.004674 \| 0.007302` | `1284 \| 0.14 \| 125.7 \| 6 \| < .001 \| 0.09 \| 61.86% \| 0.390 \| 0.004673 \| 0.007300` |

The text built on this table stays as it is (Sections 5.3.2.3, 5.4.1 and
6.1.4.3):

* residual SD 0.19 vs mean 0.16;
* one-fifth (19.52%) to about three-fifths (61.86%);
* ICCs 0.16, 0.14, 0.41, 0.39 and 0.29 (detail 0.405 still rounds to 0.41);
* "14% and 41%" and "ICC up to 0.41".

### Edit 6: Chapter 5 figures to swap in

All files are in `vizex_replication/rerun_outputs/figures/`.

* **Figure 5.3 B–F:** swap in `fig5.3_B_exp_detail.png`,
  `fig5.3_C_exp_r_directionality.png`, `fig5.3_D_exp_theta_directionality.png`,
  `fig5.3_E_exp_brightness.png`, `fig5.3_F_exp_contrast.png` and
  `fig5.3_key.png`. The 2.5 Hz column loses one point: detail mean
  0.079 → 0.078, brightness 0.729 → 0.733. Other columns are unchanged.
* **Figure 5.3A:** no change needed. The stored GPT-4o labels never included
  P358 trial 8, so χ²(18) = 548.76, 61%/39%, 96%, 82%, 0.5% and "just over
  1%" are all unchanged. `fig5.3_A_content_categories.png` is a regenerated
  copy if you want it.
* **Figure 5.1:** optional swap (`fig5.1_A…E_*.png`, `fig5.1_key.png`). It is
  visually identical: the same 296 trials, with detail and tangential values
  rescaled by less than 0.02%.
* **Figure S5.4 (heatmaps): cannot be regenerated without the images.** The
  2.5 Hz heatmap currently includes P358 trial 8 (1 of 198 images, 10% trimmed
  mean), so the difference will not be visible. To regenerate it exactly, run
  `python rerun.py --with-images` with `005_cleaned_data/` in place and use
  `rerun_outputs/figures/S5.4_heatmaps/`. Otherwise keep the current panels.
* **Figure 5.2 (random sample of 8 per condition):** the code that drew it is
  not in the repositories, so I could not check it. Please confirm by eye that
  the 2.5 Hz column does not show
  `saveTime_2510241430__…participant_P358__trialN_8__trialT_1__fileType_cropped.png`.
  If it does, replace that image with another 2.5 Hz recreation.

### Verified unchanged in Chapter 5 (no edit)

* Title, 5.2.1.4 and 5.3 counts: 1,580 / 1,186 / 98 / 296 and 99 participants.
* Section 5.3.1 / Figure 5.1 caption: slopes 0.43 to 0.93, mean 0.76, all p
  < .001.
* 10 vs 15 Hz follow-up: b = 0.071, SE = 0.021, z = 3.39, p < .001. The
  unrounded values are 0.07094, 0.02093 and 3.389, on the same 395 trials.
* Section 5.3.2.2 descriptives: peak at 15 Hz; tangential minimum at 15 Hz;
  80 Hz tangential mean 0.62; 82% Null.
* All of Section 5.3.2.1.
* Section 5.4.1: 14%–41%; validation ICCs 0 to 0.23.

---

## Chapter 6, Section 6.1

### Edit 7: Table 6.4, five rows

Columns: Coefficient | SE | z | uncorrected p | corrected p | lower CI |
upper CI.

| Row | Ctrl-F | Current | Replace with |
|---|---|---|---|
| Condition [Alpha] | `6.04` | `0.88 \| 0.15 \| 6.04 \| <0.001 \| <0.001 \| 0.59 \| 1.16` | `0.88 \| 0.15 \| 6.05 \| <0.001 \| <0.001 \| 0.59 \| 1.16` |
| Experiment 4.2 (Generative) | `-0.44` | `-0.08 \| 0.17 \| -0.44 \| 0.66 \| 1.00 \| -0.42 \| 0.26` | `-0.08 \| 0.17 \| -0.45 \| 0.66 \| 1.00 \| -0.42 \| 0.26` |
| Experiment 5.1 (Hybrid) | `1.08` | `0.17 \| 0.15 \| 1.08 \| 0.28 \| 1.00 \| -0.13 \| 0.47` | `0.16 \| 0.15 \| 1.06 \| 0.29 \| 1.00 \| -0.14 \| 0.46` |
| Interaction: Alpha × Exp 5.1 | `-0.96` | `-0.17 \| 0.17 \| -0.96 \| 0.34 \| 1.00 \| -0.51 \| 0.17` | `-0.16 \| 0.17 \| -0.94 \| 0.35 \| 1.00 \| -0.50 \| 0.18` |
| Interaction: Gamma × Exp 5.1 | `0.32 \| 0.75` | `0.06 \| 0.17 \| 0.32 \| 0.75 \| 1.00 \| -0.28 \| 0.40` | `0.06 \| 0.17 \| 0.37 \| 0.71 \| 1.00 \| -0.28 \| 0.40` |

All other rows are unchanged, including the participant variance of
0.24 (0.05). Every experiment and interaction term still has adjusted
p > .99. The gamma condition's adjusted p is still .003.

### Edit 8: Table 6.4 caption

* **Ctrl-F:** `Residual variance was 0.58 and ICC was 0.29.`
* **Replace with:** `Residual variance was 0.58 and ICC was 0.30.`
* **Why:** the ICC was 0.2949 and is now 0.2952.
* The note that the Alpha × Exp 4.2 and Gamma × Exp 4.2 values are identical
  only through rounding still holds (0.0774 vs 0.0769; z 0.411 vs 0.409); see
  optional item O7.

### Edit 9: Section 6.1.2, ICC sentence

* **Ctrl-F:** `The resulting intraclass correlation coefficient (ICC) was 0.29`
* **Current:** `The resulting intraclass correlation coefficient (ICC) was 0.29: after accounting for the fixed effects, 29% of the remaining variation was associated with participant-level clustering within experiments, while 71% was residual observation-level variation.`
* **Replace with:** `The resulting intraclass correlation coefficient (ICC) was 0.30: after accounting for the fixed effects, 30% of the remaining variation was associated with participant-level clustering within experiments, while 70% was residual observation-level variation.`

### Edit 10: Figure 6.1, swap in the new figure

* **File:** `thesis_meta_analysis/rerun_outputs/fig6.1_pooled_new.png` (panels A
  and B).
* **Panel B:** two cells change, 3.2 vs 5.1 from 0.20 to 0.21 and 4.2 vs 5.1
  from 0.24 to 0.25. The rest are unchanged (0.44, 0.07, 0.52, 0.01).
* **Panel A:** Exp 5.1 gamma mean 0.877 → 0.881; FF mean 0.290 → 0.287.
* **Caption:** unchanged. The error bars are now seeded bootstrap 95% CIs;
  see optional item O5.

### Verified unchanged in Section 6.1 (no edit)

* Table 6.3 (the thesis already shows the corrected 198/494/580/1,422), and
  "1,422 trial-level detail scores".
* Section 6.1.2 text:
  * intercept β = 0.12 [−0.12, 0.36], adj p > .99;
  * gamma β = 0.53 [0.25, 0.82], adj p = .003;
  * alpha β = 0.88 [0.59, 1.16], adj p < .001;
  * "all adjusted p > .99"; "approximately ±0.35";
  * residual variance 0.58; participant variance 0.24;
  * d = 0.01–0.52, smallest 4.1 vs 5.1 = 0.01, largest 4.1 vs 4.2 = 0.52,
    remaining 0.07–0.52;
  * gamma/FF lower in 3.2 and 4.2 than in 4.1 and 5.1.
* Section 6.1.3.2: residual 0.58.
* Section 6.1.4.1: d = 0.01–0.52 and the compressed-range sentence.
* Table 6.1, RO 1.4.

---

## Optional: pre-existing reporting issues

These were not caused by the exclusion; they are the same on the old data.
They are not needed for consistency. Fix them only if you have time.

* **O1. Figure 5.1 caption.** Ctrl-F:
  `The blue solid line indicates the ordinary least-squares fit estimated by the LME model.`
  * The code draws an OLS line of recreation on stimulus value, not the LME
    slope.
  * Suggested text: `The blue solid line shows the ordinary least-squares fit of recreation on stimulus value; the LME slopes, which account for participant, are given in Supplementary Table S5.3.`
* **O2. Section 5.3.2.2.** Ctrl-F:
  `Radial orientation followed a similar, albeit less pronounced, distribution`
  * Radial means (2.5 → 80 Hz): 0.383, 0.474, 0.446, 0.447, 0.438, 0.352,
    0.184. This is an inverted U, but it peaks at 5 Hz, not 15 Hz.
  * Suggested text: `Radial orientation followed a broadly similar, albeit less pronounced, inverted-U distribution (highest at 5–15 Hz)`.
* **O3. Section 5.3.2.2.** Ctrl-F:
  `Brightness increased with frequency, while contrast decreased`
  * Neither is strictly monotonic. Brightness is 0.768 at 10 Hz and 0.766 at
    15 Hz. Contrast is 0.158, 0.161, 0.145, 0.154 at 2.5/5/10/15 Hz.
  * Suggested text: `Brightness generally increased with frequency, while contrast generally decreased`.
* **O4. Section 6.1.1.2.** The pooled LME was fitted by REML (the statsmodels
  default); Chapters 4–5 used ML. Suggested addition after `Participant ID was included as a random intercept.`:
  `The model was fitted by restricted maximum likelihood (REML).`
  An ML fit gives participant variance 0.24, residual 0.58 and ICC 0.29, with
  coefficients identical to 1e-4, so the conclusions are unaffected.
* **O5. Figure 6.1 caption.** Ctrl-F: `Error bars denote 95% confidence intervals.`
  * These are bootstrap CIs of the trial-level mean that ignore participant
    clustering.
  * Suggested text: `Error bars denote bootstrapped 95% confidence intervals of the trial-level mean (not adjusted for repeated measures).`
* **O6. Section 5.3.2.1, χ²(18) = 548.76.**
  * 14 of the 28 expected cell counts are below 5 (minimum 0.53, in the
    sparse Semantic/Mixed columns), so the asymptotic p is approximate.
  * A seeded 20,000-permutation Monte Carlo test also gives p < .001.
  * Optional wording: `(chi^2(18) = 548.76, p < .001; Monte Carlo p < .001)`.
* **O7. Table 6.4 caption.** `the unrounded estimates diverge in the third decimal place`:
  * the coefficients are 0.0774 vs 0.0769, which are equal when rounded to
    3 dp;
  * the z values 0.411 vs 0.409 do differ at 3 dp.
  * Suggested text: `the unrounded estimates differ slightly (0.0774 vs 0.0769; z = 0.411 vs 0.409)`.
