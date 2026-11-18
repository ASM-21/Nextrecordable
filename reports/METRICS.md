# nextrecordable metrics

> **DATA NOTICE: some input files are third-party copies of OSHA's published files, not downloads from osha.gov. See the data provenance table.**

Generated 2026-11-18T03:32:35+00:00 from `ita_panel_enforced`. Trained on filings through 2020, early-stopped and calibrated on 2021, scored on 2022 (predicting 2023 outcomes). Every number below is recomputed from `reports/metrics.json` by `pipeline.render_metrics_md`.

## Data

| item | value |
|---|---|
| data provenance | third_party_copy |
| ita_summary_2019 rows | 290,475 |
| ita_summary_2020 rows | 293,391 |
| ita_summary_2021 rows | 307,902 |
| ita_summary_2022 rows | 346,799 |
| ita_summary_2023 rows | 374,662 |
| panel establishment-years | 1,588,199 |
| distinct establishments | 748,272 |
| key-years with duplicate filings (deduplicated) | 18,122 |
| establishment-years with impossible hours or headcount (kept, no rate) | 26,969 |
| establishment-years with impossible case counts (kept, no rate) | 223 |

Source files:

| file | kind | snapshot | origin |
|---|---|---|---|
| dol/OSHA_inspection.zip | official | 2026-11-15 | https://data.dol.gov/data-catalog/OSHA/inspection/OSHA_inspection.zip |
| dol/OSHA_violation.zip | official | 2026-11-15 | https://data.dol.gov/data-catalog/OSHA/violation/OSHA_violation.zip |
| ita_summary/ita_summary_2019.csv | third_party_copy | unknown (latest created_timestamp 2020-12-31) | https://github.com/nbrady387/DSCI-235-Final at 936d58068d3262ea099e031ea1ac82b5c4c918c2, file 'CY 2019.csv' |
| ita_summary/ita_summary_2020.csv | third_party_copy | unknown (latest created_timestamp 2021-12) | https://github.com/nbrady387/DSCI-235-Final at 936d58068d3262ea099e031ea1ac82b5c4c918c2, file 'CY 2020.csv' |
| ita_summary/ita_summary_2021.csv | third_party_copy | through 2022-08-29 | https://github.com/nbrady387/DSCI-235-Final at 936d58068d3262ea099e031ea1ac82b5c4c918c2, file 'CY 2021 submitted thru 8-29-2022.csv' |
| ita_summary/ita_summary_2022.csv | third_party_copy | unknown (latest created_timestamp 2023-12) | https://github.com/nbrady387/DSCI-235-Final at 936d58068d3262ea099e031ea1ac82b5c4c918c2, file 'CY 2022.csv' |
| ita_summary/ita_summary_2023.csv | third_party_copy | through 2024-03-16 | https://github.com/nbrady387/DSCI-235-Final at 936d58068d3262ea099e031ea1ac82b5c4c918c2, file 'ITA 300A Summary Data 2023 through 3-16-2024.csv' |

ITA 300A files for 2019-2023 are copies of OSHA's published CSVs taken from a public GitHub repository (a 2024 student project), because osha.gov refused connections from the build machine. They have not been compared with OSHA's own files. The 2023 file is an early snapshot (through 2024-03-16), so late 2023 filers are missing. DOL enforcement files are official bulk downloads from data.dol.gov.

## Binary model: any recordable case next year

| metric | model | size + NAICS baseline |
|---|---|---|
| AUC | 0.859 | 0.813 |
| share of next-year cases in top 10% | 49.8% | 44.7% |
| share of next-year DART cases in top 10% | 49.0% | 44.7% |
| Brier score (always-base-rate: 0.236) | 0.147 | |
| Brier skill vs base rate | 37.4% | |
| expected calibration error, holdout year | 0.011 | |
| top-10% lift (ceiling = 1 / base rate) | 1.588 (ceiling 1.614) | |

Base rate: 62.0% of establishments logged a case.

### Split by establishment

Every establishment that filed in a training year and again in the holdout year is in both. Scoring only establishments the model never saw separates prediction from recognition.

| holdout rows | n | base rate | AUC | baseline AUC | Brier |
|---|---|---|---|---|---|
| establishments seen in training/validation | 134,984 | 63.8% | 0.865 | 0.815 | 0.143 |
| establishments unseen in training/validation | 63,978 | 58.1% | 0.846 | 0.805 | 0.158 |

### Reliability on the holdout year

Calibrator fitted on the validation year; ten equal-count bins of the holdout year.

| bin | n | mean predicted | observed | gap |
|---|---|---|---|---|
| 1 | 19,896 | 0.097 | 0.093 | 0.004 |
| 2 | 19,896 | 0.265 | 0.262 | 0.003 |
| 3 | 19,896 | 0.378 | 0.367 | 0.011 |
| 4 | 19,896 | 0.497 | 0.480 | 0.017 |
| 5 | 19,897 | 0.618 | 0.599 | 0.019 |
| 6 | 19,896 | 0.735 | 0.716 | 0.019 |
| 7 | 19,896 | 0.837 | 0.825 | 0.012 |
| 8 | 19,896 | 0.923 | 0.907 | 0.016 |
| 9 | 19,896 | 0.969 | 0.963 | 0.006 |
| 10 | 19,897 | 0.980 | 0.984 | 0.004 |

## Rate model: expected cases per 200,000 hours next year

Judged on realized next-year cases over next-year hours. **Served rate: the credibility estimate.** The boosted model did not beat it on the holdout year, so reports and scores use the credibility estimate and show no rate drivers.

| metric | boosted model | credibility baseline | industry rate |
|---|---|---|---|
| Poisson deviance (lower is better) | 2.718 | 2.539 | 4.120 |
| Spearman vs realized rate | 0.424 | 0.441 | 0.271 |
| top-10% pooled rate / overall | 2.859 | 2.802 | 2.048 |
| predicted / actual cases | 1.228 | 1.162 | 1.182 |
| share of next-year cases in top 10% by expected cases | 60.3% | 60.3% | 55.1% |

Largest sites first (rank by hours worked) puts 51.9% of next-year cases in the top 10%.

Deviance skill of the boosted model vs credibility baseline: -7.1%; vs industry rate: 34.0%.

## Matching

DOL inspections opened 2019 to 2023 linked to an ITA filer: 7.6% of 341,919 (exact key 21,206, fuzzy 4,940, ambiguous and left unlinked 411). Most inspections are of sites that never file with ITA (construction, under 20 employees), so this share is low by design. Cross-year link: 771,257 establishment-years have last year's filing.

## SST positions (CY 2023 selection year)

Counts of establishments by position against CPL 02-01-067. Positions, not inspection lists: OSHA publishes no DART cutoffs and three of four lists are random samples.

| item | value |
|---|---|
| filed for CY 2023 (file snapshot: through 2024-03-16) | 368,726 |
| compared (20+ employees, general industry, usable rate, not government) | 232,562 |
| meet both upward-trend conditions as this product reads them | 3,730 |
| filed CY 2022 with 20+ employees, no CY 2023 filing found | 98,652 |
| establishments with 20+ employees not found the next year, by year | 2020: 32.2%, 2021: 30.0%, 2022: 33.3%, 2023: 40.7% |
| manufacturing: n, median / p75 / p90 / p95 DART | 54,629: 1.16 / 3.25 / 6.00 / 8.43 |
| non_manufacturing: n, median / p75 / p90 / p95 DART | 177,933: 1.53 / 4.38 / 8.34 / 12.22 |

## Against the synthetic baseline

The same code on the synthetic fixtures (`synthetic_baseline.json`, generated 2026-11-18T03:20:53+00:00). Synthetic data was built to carry signal, so a drop here is expected and is the honest number.

| metric | real | synthetic | change |
|---|---|---|---|
| binary AUC | 0.859 | 0.842 | +0.018 |
| binary AUC, size + NAICS baseline | 0.813 | 0.806 | +0.007 |
| binary AUC, establishments unseen in training | 0.846 | n/a | n/a |
| Brier score | 0.147 | 0.096 | +0.052 |
| Brier skill vs base rate | 37.4% | 19.0% | +18.4 pts |
| expected calibration error (holdout) | 0.011 | 0.019 | -0.008 |
| share of next-year cases in top 10% (binary) | 49.8% | 29.1% | +20.8 pts |
| rate deviance skill vs credibility | -7.1% | 4.3% | -11.4 pts |
| rate deviance skill vs industry | 34.0% | 61.2% | -27.2 pts |

## Sanity gates

| gate | status | value | rule |
|---|---|---|---|
| years_loaded | pass | 5 | >= 4 years (fail < 3) |
| rows_per_year | pass | 290,475 | >= 1,000 rows in every year |
| zip5_valid_share | pass | 1.000 | >= 0.98 (fail < 0.90) |
| leading_zero_zips_kept | pass | 0.994 | >= 0.95 of CT/MA/ME/NH/NJ/PR/RI/VT ZIPs start with 0 |
| hours_per_employee_median | pass | 1,810.170 | 1,200 to 2,600 hours (fail outside 500 to 4,000) |
| hours_per_employee_outliers | pass | 0.013 | <= 2% outside 100 to 8,760 hours per employee |
| duplicate_key_years | pass | 0.011 | <= 5% of key-years had more than one filing (fail > 15%) |
| year_filing_for_vs_file_year | pass | 0.000 | <= 5% of rows report a different year than their file |
| hard_id_split_across_keys | pass | 0.000 | <= 2% of establishment_ids map to more than one key in a year |
| size_code_harmonized | pass | 0 | no 21/22 size codes after harmonization |
| target_base_rate | pass | 0.652 | 0.20 to 0.95 |
| cross_year_link_share | pass | 0.486 | >= 30% of rows have last year's filing |
| inspection_link_share | warn | 0.076 | >= 15% of inspections in panel years link to an ITA filer |
