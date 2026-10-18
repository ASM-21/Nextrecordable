# nextrecordable

Forward-looking injury risk for any U.S. establishment that files OSHA injury logs: its expected recordable case rate next year, where that rate sits among industry peers, and what drives it. Built entirely on public data (OSHA's Injury Tracking Application and DOL enforcement records) for workers' comp brokers, loss-control engineers, and safety managers.

[![CI](https://github.com/ASM-21/nextrecordable/actions/workflows/ci.yml/badge.svg)](https://github.com/ASM-21/nextrecordable/actions/workflows/ci.yml)

> **Status.** Runs end to end on synthetic data built to reproduce the real files' quirks. It has not yet been run on real OSHA data; that first run is one command (`nextrecordable run-all`) and writes `reports/METRICS.md`. Until that report exists, no accuracy claim about real establishments should be made from this repo.

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

## Method

| piece | how |
|---|---|
| Data | ITA Form 300A summaries (2016 onward, hundreds of thousands of establishments a year) and DOL `osha_inspection` / `osha_violation`. Every CSV column loads as text; typing happens explicitly in `schema.py`. |
| Establishment identity | `estab_key` = hash of normalized name, state and 5-digit ZIP. One Python implementation of every normalization rule (`normalize.py`), joined into SQL rather than re-implemented in it. |
| Panel | one row per establishment and year; lags and next-year outcomes only across adjacent calendar years; amended filings deduplicated; NAICS-4 median and pooled rates per year. |
| Rate model | LightGBM, Poisson objective, offset log(hours) + log(experience-rated rate). The trees learn corrections to the actuarial estimate; early stopping on a held-out year decides how many survive. |
| Binary model | LightGBM, P(any recordable next year), isotonic-calibrated on the validation year. |
| Validation | temporal: train on years up to T, early-stop and calibrate on T+1, report on T+2. |
| Matching | fixed-weight Fellegi-Sunter scorer in DuckDB SQL (name, numbers in the name, ZIP, street). Used for lookups, batch uploads, and linking DOL inspections to ITA filers. Close calls come back `ambiguous` with suggestions instead of a guess. |

## Why you can trust the numbers it will produce

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

The match harness (`tests/test_match_quality.py`) pins floors: exact, uppercase, abbreviated, ZIP+4 and lost-leading-zero lookups 100%; one typo at least 97%; zero wrong matches in every case; a changed plant number or a similar name in the same ZIP never matched.

## Quick start

```bash
git clone https://github.com/ASM-21/nextrecordable && cd nextrecordable
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # synthetic data, no network

nextrecordable run-all         # download ITA files, load, build, sanity gates, train
nextrecordable serve           # lookup page and API on http://127.0.0.1:8000
```

Optional enforcement features: download `osha_inspection` and `osha_violation` CSVs from the DOL enforcement data catalog (enforcedata.dol.gov, moving to data.dol.gov) into `data/raw/dol/` before `run-all`, or load them with `nextrecordable load-enforcement --inspection ... --violation ...`.

Step by step: `nextrecordable download summary`, `load`, `load-enforcement`, `build`, `sanity`, `train`, `serve`, plus `explore` and `query "SQL"`. Paths default to `./data` and `./reports`; override with `NEXTRECORDABLE_DATA_DIR`, `NEXTRECORDABLE_DB_PATH`, `NEXTRECORDABLE_MODEL_DIR`, `NEXTRECORDABLE_REPORTS_DIR`.

### API

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" \
  -d '{"name": "Acme Forge Inc", "state": "IL", "zip_code": "61602", "street_address": "5 Mill Rd"}'

curl -X POST localhost:8000/score-batch -F "file=@book_of_business.csv"
```

Batch CSVs need name, state and ZIP columns; common headers (`company`, `zip`, `address`, `st`) are recognized, and ZIPs that lost their leading zero in Excel are repaired. Limits: 5 MB and 5,000 rows per upload. Unmatched and ambiguous rows are returned separately, never silently dropped. Run one server process per database file.

## Read before relying on a score

- OSHA injury logs undercount recordable cases; published estimates of the share missed run from roughly 30 to 70 percent, depending on industry and method. Scores predict reported cases, which also reflect how carefully an employer records them.
- Only establishments required to file with the ITA appear; most workplaces under 20 employees don't file.
- Training rows need a next-year filing, so establishments that stop filing (closures, falling below thresholds, non-compliance) are underrepresented.
- Names change; one site can split into two records across years, and two sites sharing a name and ZIP are combined.
- Zero inspections means none on record, not inspected and clean.
- Do not republish scores about named establishments without your own legal review.

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
  api.py         FastAPI app; static/index.html is the lookup page
  cli.py         the nextrecordable command
tests/           synthetic ITA and DOL generators plus the suite
```

See `CONTRIBUTING.md` for conventions and good first contributions, and `ROADMAP.md` for what's next.
