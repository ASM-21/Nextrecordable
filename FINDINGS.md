# Findings from the first real-data run

November 2026. ITA Form 300A summaries for calendar years 2019 to 2023 and DOL enforcement records through November 15, 2026. **The ITA files are third-party copies of OSHA's published CSVs, not downloads from osha.gov** (see `DATA-GAPS.md`); every output says so until official files replace them.

Numbers here come from `reports/metrics.json` and `reports/falsification.json`. `tests/test_docs.py` checks each one against those files, so this page cannot drift from the results. Regenerate with `nextrecordable train`, `nextrecordable falsify`.

## Headline, and how far to trust it

| claim | number | trust |
|---|---|---|
| Binary model ranks "any recordable case next year" | AUC 0.859 vs 0.813 for a size + NAICS model | survived every check below |
| Probabilities are calibrated on a year the calibrator never saw | expected calibration error 0.011; Brier 0.147 vs 0.236 for always predicting the base rate | good |
| The boosted rate model improves on experience rating | **no**: deviance skill -7.1% vs the credibility estimate | product now serves the credibility estimate |
| The credibility estimate improves on the industry rate | deviance skill of the boosted model vs industry 34.0%; the credibility estimate is better still | good, and it is plain actuarial math |

The binary model is mostly a measure of exposure plus last year's record. That is useful for triage (the top 10% of establishments by predicted probability hold 49.8% of next year's cases, against 44.7% for the size + NAICS model), but it is not a safety-culture detector.

## Attempts to break the headline

### 1. Is it just size? No.

Inside every employee-count band the model beats the size + NAICS model:

| employees | AUC, model | AUC, size + NAICS |
|---|---|---|
| 1-19 | 0.777 | 0.749 |
| 20-49 | 0.733 | 0.640 |
| 50-99 | 0.788 | 0.673 |
| 100-249 | 0.838 | 0.707 |
| 250-999 | 0.899 | 0.766 |
| 1,000+ | 0.927 | 0.816 |

Simpler rules for comparison, on the whole holdout year: rank by hours worked 0.777, "had a case this year" 0.745, "number of cases this year" 0.815. The last one is the baseline a skeptic should quote. Plain persistence alone does as well as size + NAICS, so the model's margin is over "same as last year", not just over "big sites log cases".

### 2. Is it recognition of establishments seen in training? Mostly no.

Two thirds of holdout rows belong to establishments that also appear in training years. Scoring only establishments never seen in training or validation, by name-state-ZIP key and by OSHA establishment_id (which catches renames and moves): n = 62,033, AUC 0.846 vs 0.806 for the size + NAICS model. Only 1,945 holdout establishments had a new key but an old establishment_id, so the coarse key rarely splits one site into two.

### 3. Is the holdout year special? No.

Retraining with 2021 as the test year (train 2019, validate 2020) gives AUC 0.872 vs 0.832. The boosted rate model ties the credibility estimate there (deviance skill 0.0%) and loses in 2022, so it is not served.

### 4. Leakage across years

The panel only takes lags and next-year outcomes across adjacent calendar years, enforcement features use inspections opened in years t-1 and t, and validation and test years are strictly later than training years (pinned by existing tests). The establishment check above is the leakage test that matters on real data: restricting to never-seen establishments lowered AUC only from 0.859 to 0.846.

### 5. Tiny denominators

The rate model loses to the credibility estimate in every hours band, worst at the extremes: -16.8% under 40,000 hours (about 20 full-time workers) and -23.1% above 1,000,000 hours. Sites under 20 FTE hold 3.1% of next year's cases.

For the SST report this matters more than for the model. Among establishments in the top 10% of their SST group by CY 2023 DART rate, 19.4% (manufacturing) and 20.3% (non-manufacturing) got there with two or fewer DART cases, and 49.1% and 56.8% work under 100,000 hours a year. For a 50-FTE manufacturer, about 2.4 more DART cases move it from the group median to the 90th percentile. The report states the case counts next to every rate so a reader can see this.

### 6. Data problems found and fixed on the way (each now a test)

- Line breaks inside quoted address fields stopped the CSV load.
- One year file writes every number with two decimals, identifiers included, so a year's establishments never linked to the years around them.
- Four timestamp spellings meant "latest amendment wins" was decided by string order.
- 26,969 establishment-years report hours that cannot be true for their headcount (one seven-person site reported 16.8 trillion hours, which alone pulled the pooled 2019 case rate for all filers from 4.0 to 0.02). They keep their counts but get no rate.
- 223 establishment-years report impossible case counts (three filings of 1,923 cases at sites of 25 to 218 people); they get no rate and are never a training label.
- DOL's bulk files are chunked pandas exports with "nan" and "NaT" placeholders.

## Matching

7.6% of DOL inspections opened 2019 to 2023 link to an ITA filer. The sanity gate warns below 15%, but most inspections are of construction sites and small employers that never file with ITA, so a low share is expected; the gate threshold was set before real data existed. Not yet measured: precision of the fuzzy links on real data (no labeled pairs).

## Missing filings are mostly churn, not non-compliance

Among establishments with 20 or more employees outside construction, the share that filed one year and is not found under the same name, state and ZIP the next was 32.2% (2020), 30.0% (2021), 33.3% (2022) and 40.7% (2023). The 2023 figure includes the early snapshot. The SST report therefore never calls an establishment a non-responder; it says no filing was found and gives this background rate.

## What I could not verify

- The ITA copies against OSHA's own files (osha.gov refused this machine).
- Whether OSHA reads "twice the private sector national average" as a DART rate or a total case rate. The product states its reading (DART, BLS 2022 private industry 1.7, so 3.4) as its own.
- The BLS figure itself was read from BLS state releases that reproduce the national table, not from the national release.
- Whether a successor SST directive has been issued; none was found by search on 2026-11-16.
