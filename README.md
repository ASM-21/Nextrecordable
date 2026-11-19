# nextrecordable

Forward-looking injury risk for any U.S. establishment that files OSHA injury logs: its expected recordable case rate next year, where that rate sits among industry peers, and what drives it. Built entirely on public data (OSHA's Injury Tracking Application and DOL enforcement records) for workers' comp brokers, loss-control engineers, and safety managers.

For manufacturers it answers a narrower question: **how does OSHA's Site-Specific Targeting program see us?** The SST exposure report places one establishment against the published criteria of OSHA's current targeting directive (CPL 02-01-067), using the same Form 300A data OSHA uses.

[![CI](https://github.com/ASM-21/nextrecordable/actions/workflows/ci.yml/badge.svg)](https://github.com/ASM-21/nextrecordable/actions/workflows/ci.yml)

> **Status (November 2026).** Trained and evaluated on real data: ITA Form 300A summaries for 2019 to 2023 and DOL enforcement records through November 2026. The ITA files used are **third-party copies** of OSHA's published CSVs, because osha.gov was unreachable from the build machine; every output says so until `DATA-GAPS.md` is followed. Results: `reports/METRICS.md`; attempts to break them: `FINDINGS.md`.

An independent project. Not affiliated with, endorsed by, or produced on behalf of OSHA or the U.S. Department of Labor.

## What you get

For one establishment (lookup page at `/` or `POST /score`) or a whole book of business (`POST /score-batch` with a CSV):

| output | meaning |
|---|---|
| `expected_tcr_next` | expected recordable cases per 100 full-time workers next year (per 200,000 hours). Size-adjusted; batch results rank by it |
| `peer_pctile`, `risk_band` | where that rate falls among establishments in the same NAICS group |
| `experience_rated_tcr_next` | the starting point: the establishment's last two years blended with its industry's rate |
| `risk_drivers` | which features moved the estimate off that starting point, as rate multipliers |
| `p_any_next` | chance of at least one recordable case next year (calibrated). Mostly a function of headcount; use the rate to compare sites |
| `history`, `enforcement` | filed injury counts and rates by year; OSHA inspections, citations and initial penalties in the last two years |
| `caveats` | always included, see below |
| `data_notice` | null when the loaded files are OSHA's own; otherwise says they are third-party copies, undeclared or synthetic |

`expected_tcr_next` comes from the boosted rate model only when that model beat the experience-rated estimate on the holdout year. On the current data it did not, so the experience-rated estimate is served and `risk_drivers` are not shown (`rate_served` in `/health`).

## SST exposure report

`POST /sst-exposure` (JSON) and `POST /sst-exposure-pdf` take the same body as `/score` and return, for one establishment:

- DART rate and case counts for every filed year, with the industry median;
- position among CY 2023 filers in its SST group (manufacturing, or non-manufacturing excluding construction) and in its NAICS group, as the share of peers with a lower rate;
- where it sits against each of the directive's four lists (high-rate, upward-trending, low-rate, non-responders), each next to the directive's own words;
- whether a CY 2023 filing was found, with the year-over-year churn that makes a missing filing ambiguous;
- inspections on record and the 36-month deletion rule.

`POST /sst-batch` takes a CSV of establishments (an extension center's client plants, a broker's book) and returns one row of SST position per establishment, as JSON or, with `?format=csv`, as a spreadsheet whose first line is the statement below. It counts each row against the plan's batch limit.

Every report carries this sentence: **This report describes position against published criteria and never predicts whether OSHA will inspect.** OSHA does not publish the DART cutoffs it uses, three of the four lists are random samples, and only OSHA and State Plans see the lists, so the report gives percentile position and states no cutoff. The directive's dates, quotes and source are in `nextrecordable/data/sst_cpl_02_01_067.json`; no successor directive had been announced as of November 16, 2026. No Form 300/301 case data, and so no narrative about any worker, can reach the report.

## Results on real data

Holdout year 2022 (predicting 2023), trained through 2020. Full tables in `reports/METRICS.md`; checks in `FINDINGS.md`.

| | model | baseline |
|---|---|---|
| AUC, any recordable case next year | 0.859 | 0.813 (size + NAICS) |
| AUC, establishments never seen in training | 0.846 | 0.806 |
| Brier score | 0.147 | 0.236 (always predict the base rate) |
| expected calibration error, holdout year | 0.011 | |
| next year's cases in the top 10% | 49.8% | 44.7% |
| boosted rate model, deviance skill vs experience rating | -7.1% (so it is not served) | |

What survived: the binary model's edge holds inside every size band, for establishments never seen in training, and on a second test year. What did not: the boosted rate model, which loses to plain experience rating. Matching links 7.6% of DOL inspections opened 2019 to 2023 to an ITA filer; most inspections are of sites that never file.
## Method

| piece | how |
|---|---|
| Data | ITA Form 300A summaries (2016 onward, hundreds of thousands of establishments a year) and DOL `osha_inspection` / `osha_violation`. Every CSV column loads as text; typing happens explicitly in `schema.py`. Filings with impossible hours or case counts keep their counts but get no rate. `data/raw/SOURCES.json` declares where each file came from. |
| Establishment identity | `estab_key` = hash of normalized name, state and 5-digit ZIP. One Python implementation of every normalization rule (`normalize.py`), joined into SQL rather than re-implemented in it. |
| Panel | one row per establishment and year; lags and next-year outcomes only across adjacent calendar years; amended filings deduplicated; NAICS-4 median and pooled rates per year. |
| Rate model | LightGBM, Poisson objective, offset log(hours) + log(experience-rated rate). The trees learn corrections to the actuarial estimate; early stopping on a held-out year decides how many survive, and the holdout year decides whether the corrections are served at all. |
| Binary model | LightGBM, P(any recordable next year), isotonic-calibrated on the validation year. |
| Validation | temporal: train on years up to T, early-stop and calibrate on T+1, report on T+2. |
| Matching | fixed-weight Fellegi-Sunter scorer in DuckDB SQL (name, numbers in the name, ZIP, street). Used for lookups, batch uploads, and linking DOL inspections to ITA filers. Close calls come back `ambiguous` with suggestions instead of a guess. |

## Why you can trust the numbers

- **Every model faces the baseline a skeptic would propose.** The rate model is scored against the industry rate and against the credibility blend it starts from; the binary model against a size + NAICS model. `reports/METRICS.md` prints the model and the baselines side by side, so "the ML adds nothing here" is a visible outcome, not a hidden one.
- **Metrics that can't flatter.** Top-decile lift is printed next to its ceiling (1 / base rate: with 75% of sites logging a case, no model can exceed 1.33). Capture rate (share of next year's cases in the top 10%) and Poisson deviance on next-year hours carry the conclusions.
- **Data gates before training.** `nextrecordable sanity` checks ZIP validity, that leading-zero ZIPs survived loading, hours per employee, duplicate filings, `year_filing_for` drift, hard-ID consistency, the size-code split, cross-year linkage, base rate, and the DOL link rate. A failed gate stops `run-all`.
- **Every bug found so far is a regression test.** Each was reproduced before it was fixed:

| bug (version before 0.2.0) | evidence | test |
|---|---|---|
| Categorical codes came from Python's per-process `hash()` | serving from a new process changed 105 of 200 scores | `test_scores_identical_in_a_fresh_process` |
| Shared connection and match tables across requests | 120 concurrent lookups: 18 wrong establishments, 100 errors | `test_concurrent_lookups_return_the_right_establishment` |
| ZIP typed as an integer after DuckDB's sample | CT's 06511 stored as 6511 and never found | `test_leading_zero_zip_survives_type_sniffing` |
| Amended filings: two rows per establishment-year | next-year target was the same year's other filing | `test_amended_filing_does_not_leak_into_targets` |
| Skipped filing years | lag1 two years old, target two years ahead | `test_skipped_year_gives_no_lag_and_no_target` |
| DOL letter codes read as words | every severity and inspection-type feature zero; deleted citations counted | `test_real_dol_codes_are_counted` |
| Matcher trained per request | one-typo lookups matched 12 of 40 | `tests/test_match_quality.py` |
| Unbounded dependency (splink 5.0) | every fresh install crashed on lookup | splink removed; major versions bounded; weekly CI against latest |
| Real files: line breaks inside quoted addresses | the 2019 load stopped at line 4,151 | `test_line_break_inside_a_quoted_field_loads` |
| Real files: one year writes IDs as `439896.00` | that year never linked to its neighbors | `test_two_decimal_export_does_not_change_ids_sizes_or_years` |
| Real files: four timestamp spellings | latest amendment chosen by string order | `test_latest_amendment_wins_in_every_timestamp_format` |
| Real files: impossible hours | one filing pulled the pooled 2019 case rate from 4.0 to 0.02 | `test_impossible_hours_cannot_enter_a_rate` |
| Real files: impossible case counts | the three largest holdout errors were keying mistakes | `test_impossible_case_count_is_not_a_rate_or_a_target` |
| DOL bulk files: pandas placeholders | a "nan" penalty made penalty sums NaN | `test_dol_bulk_layout_chunks_and_missing_tokens` |

The match harness (`tests/test_match_quality.py`) pins floors: exact, uppercase, abbreviated, ZIP+4 and lost-leading-zero lookups 100%; one typo at least 97%; zero wrong matches in every case; a changed plant number or a similar name in the same ZIP never matched.

## Quick start

```bash
git clone https://github.com/ASM-21/nextrecordable && cd nextrecordable
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # synthetic data, no network

nextrecordable run-all         # download ITA files, load, build, sanity gates, train
nextrecordable falsify         # size bands, unseen establishments, other test years -> reports/falsification.json
nextrecordable serve           # lookup page and API on http://127.0.0.1:8000
```

Optional enforcement features: download `OSHA_inspection.zip` and `OSHA_violation.zip` from data.dol.gov (exact URLs in `DATA-GAPS.md`) and unzip them into `data/raw/dol/inspection/` and `data/raw/dol/violation/` before `run-all`, or load them with `nextrecordable load-enforcement --inspection ... --violation ...`. Declare where every raw file came from in `data/raw/SOURCES.json` (format in `provenance.py`); without it, outputs are marked undeclared.

Step by step: `nextrecordable download summary`, `load`, `load-enforcement`, `build`, `sanity`, `train`, `report`, `falsify`, `serve`, plus `synthetic-baseline` (the comparison run in METRICS.md), `explore` and `query "SQL"`. Paths default to `./data` and `./reports`; override with `NEXTRECORDABLE_DATA_DIR`, `NEXTRECORDABLE_DB_PATH`, `NEXTRECORDABLE_MODEL_DIR`, `NEXTRECORDABLE_REPORTS_DIR`.

### API

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" \
  -d '{"name": "Acme Forge Inc", "state": "IL", "zip_code": "61602", "street_address": "5 Mill Rd"}'

curl -X POST localhost:8000/score-batch -H "X-API-Key: $KEY" -F "file=@book_of_business.csv"

curl -X POST localhost:8000/sst-exposure-pdf -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"name": "Acme Forge Inc", "state": "IL", "zip_code": "61602"}' -o sst-exposure.pdf
```

Batch CSVs need name, state and ZIP columns; common headers (`company`, `zip`, `address`, `st`) are recognized, and ZIPs that lost their leading zero in Excel are repaired. Limits: 5 MB and 5,000 rows per upload. Unmatched and ambiguous rows are returned separately, never silently dropped. Run one server process per database file.

### Keys, tiers and pricing

| tier | price | lookups a month | batch rows per upload | SST exposure PDF |
|---|---|---|---|---|
| Free | $0 | 100 | none | on screen only |
| Pro | $79/mo | 2,000 | 1,000 | yes |
| Team | $499/mo | 25,000 | 5,000 | yes |
| Enterprise | contact | custom | 5,000 | yes |

Tiers live in `nextrecordable/data/tiers.json`; the pricing page (`/pricing`), the API limits and this table all follow it, and a test pins them. Lookups without a key are allowed, 20 a day per client. Keys arrive by email after a Stripe Payment Link checkout (`/stripe/webhook`); email is dry-run by default and writes to `data/outbox/`. Local `nextrecordable serve` runs with accounts on; tests and `create_app` without an account store run open.

## Deploying

`Dockerfile` plus `deploy/docker-compose.yml` (the app behind Caddy, which obtains and renews TLS certificates) for one small host. `deploy/RUNBOOK.md` has the launch checklist in order (data, legal read, domain, host, DNS, Stripe test mode, email, Stripe live) and day-to-day operations.

## Read before relying on a score

- OSHA injury logs undercount recordable cases; published estimates of the share missed run from roughly 30 to 70 percent, depending on industry and method. Scores predict reported cases, which also reflect how carefully an employer records them.
- Only establishments required to file with the ITA appear; most workplaces under 20 employees don't file.
- Training rows need a next-year filing, so establishments that stop filing (closures, falling below thresholds, non-compliance) are underrepresented.
- Names change; one site can split into two records across years, and two sites sharing a name and ZIP are combined.
- Zero inspections means none on record, not inspected and clean.
- Do not republish scores about named establishments without your own legal review.
- The SST report describes position against published criteria and never predicts whether OSHA will inspect. "No filing found" is not a finding of non-compliance: in complete years about a third of filers are not found under the same name and address the next year.

## Data and license

ITA and DOL enforcement data are U.S. federal government works (public domain in the U.S.) and are downloaded at run time, not redistributed here. Code is Apache-2.0; see `LICENSE` and `NOTICE`. Cite with `CITATION.cff`.

## Layout

```
nextrecordable/
  config.py      download URLs, paths
  download.py    cached downloads, written atomically
  load.py        CSV -> DuckDB, every column as text
  schema.py      canonical typed view across release years
  normalize.py   names, addresses, ZIPs (single source of truth)
  keys.py        establishment keys, shared by ITA and DOL
  features.py    the establishment-year panel
  enforcement.py DOL inspections and citations, linking, panel windows
  match.py       fixed-weight record linkage
  models/        binary and rate models, metrics, persistence, scores
  report.py      per-establishment reports, vectorized batch scoring
  sanity.py      data gates
  pipeline.py    build, train, reports/METRICS.md
  provenance.py  where the loaded files came from; the notice every output carries
  sst.py         SST exposure report and PDF; data/sst_cpl_02_01_067.json is the directive
  falsify.py     attempts to break the headline metrics
  accounts.py    API keys, tiers (data/tiers.json), monthly metering
  billing.py     Stripe webhook: signature check, accounts, key email
  mailer.py      email behind an interface, dry-run by default
  api.py         FastAPI app; static/index.html is the lookup page
  cli.py         the nextrecordable command
tests/           synthetic ITA and DOL generators plus the suite
deploy/          docker-compose.yml, Caddyfile, env.example, RUNBOOK.md
```

`DECISIONS.md` logs the judgment calls behind all of the above; `DATA-GAPS.md` says how to replace the third-party ITA copies with official files.

See `CONTRIBUTING.md` for conventions and good first contributions, and `ROADMAP.md` for what's next.
