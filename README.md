# osha-risk

> Establishment-level workplace injury risk intelligence from OSHA's Injury Tracking Application.

[![CI](https://github.com/USERNAME/osha-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/osha-risk/actions/workflows/ci.yml)

A transparent, reproducible pipeline that turns OSHA's public Form 300A and 300/301 data into a usable risk-scoring layer for workers' comp brokers, loss-control engineers, and anyone else who needs a forward-looking view of injury risk at a specific establishment.

Working on this repo? See `CONTRIBUTING.md` for setup, conventions, and where things live.

## Status

Data pipeline, panel construction, enforcement integration, entity resolution, and baseline model all work end-to-end against a synthetic fixture (56 passing tests). Not yet run against real OSHA/DOL data. No API or UI yet — see Roadmap.

## What this does (eventually)

Given a CSV of business names and addresses (a broker's book of business, a procurement supplier list, a journalist's investigation target list), return for each:

- Estimated probability of any recordable injury in the coming year
- Estimated DART rate and TCR
- Industry-and-size-adjusted benchmark percentile
- OSHA inspection and citation history
- Driver explanations (which features pushed risk up or down)

## What this does today

```
osha_risk/
├── config.py        # OSHA download URLs, paths, schema candidates
├── download.py      # Streaming HTTP downloader with caching
├── load.py          # Unzip + load yearly CSVs into DuckDB
├── normalize.py     # Name and address normalization
├── schema.py        # Resolves column-name drift across years -> canonical view
├── features.py      # Builds establishment-year panel with lags + targets
├── enforcement.py   # DOL inspection/violation/penalty integration
├── match.py         # Splink-based fuzzy entity resolution for CSV uploads
├── models/
│   └── baseline.py  # LightGBM classifier, temporal holdout, isotonic calibration
├── report.py        # Per-establishment risk report (the API/PDF unit)
├── api.py           # FastAPI service: POST /score, POST /score-batch
└── cli.py           # Typer CLI
```

## Running the API

Train and save a model once (offline), then serve it:

```bash
make download && make load          # get real OSHA data
python -c "
from osha_risk.load import connect
from osha_risk.schema import build_canonical_view
from osha_risk.features import build_establishment_key_view, build_panel
from osha_risk.models.baseline import train_baseline, save_model

con = connect()
build_canonical_view(con)
build_establishment_key_view(con)
build_panel(con)
result = train_baseline(con, train_through=2022)  # val=2023, test=2024
save_model(result, 'data/model')
"

make serve   # OSHA_RISK_SERVE=1 uvicorn osha_risk.api:app
```

Then:

```bash
# Single establishment
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{
  "name": "Acme Manufacturing Inc",
  "street_address": "123 Main St",
  "state": "IL",
  "zip_code": "60601"
}'

# CSV batch (broker's book of business)
curl -X POST localhost:8000/score-batch -F "file=@book_of_business.csv"
# CSV needs columns: name, state, zip_code (street_address, city optional)
```

Both endpoints run the full pipeline live: entity resolution (splink + softkey fallback) against whatever `OSHA_RISK_PANEL_TABLE` points to, then `report.risk_report()` for the calibrated score, drivers, and history. See `osha_risk/api.py` docstring for the concurrency caveat (requests are serialized behind a lock — fine for a demo, not for production load).

## Quick start

```bash
git clone <repo-url>
cd osha-risk
python -m venv .venv && source .venv/bin/activate
make dev-install
make test          # confirm everything works before touching real data

# Download + load real OSHA data
make download       # ITA 300A summary, 2016-2024
make load            # unzip + load into DuckDB
make explore         # sanity-check row counts, schema, top NAICS
```

See `Makefile` for all commands, or `CONTRIBUTING.md` for repo conventions
and where to look when extending the pipeline.

## Run the test suite

```bash
make test      # verbose
make test-fast # quiet, stop on first failure
```

The tests use synthetic OSHA-shaped fixtures (`tests/synthetic.py`,
`tests/synthetic_enforcement.py`) — establishments with intentional pre-2023
/ post-2023 schema drift and independently-varying injury and enforcement
signals — and exercise the full pipeline from CSV load through model
training and risk report generation. No network access or real data
required to run tests.

## Data sources

| Source | What | Coverage | URL |
|---|---|---|---|
| OSHA ITA 300A | Establishment-year injury summaries | 2016 - 2024 | [osha.gov ITA data](https://www.osha.gov/Establishment-Specific-Injury-and-Illness-Data) |
| OSHA ITA 300/301 | Case-level details with narratives | 2023 - 2024 | same |
| DOL Enforcement | Inspections, citations, penalties | 1973 - present | [enforcedata.dol.gov](https://enforcedata.dol.gov/views/data_catalogs.php) |
| BLS QCEW | Employment by establishment | quarterly | (not yet integrated) |

## Methodological honesty

This is the section that will be longest in the final docs. A few things users and reviewers should know:

**Underreporting.** Multiple academic studies and BLS analyses estimate that 30 to 70 percent of OSHA-recordable injuries go unreported, with rates varying by industry. A model trained on this data partly learns reporting behavior. Treat predicted rates as "predicted REPORTED rates" unless you have independent reason to believe coverage is uniform.

**Entity resolution is soft.** Establishment IDs are only present in recent years. Older years are matched on normalized (name, address) hashes, with expected leakage of 5 to 15 percent (same establishment appearing as two keys, or two establishments collapsing into one). Match quality metrics are produced as part of the pipeline.

**Coverage gaps.** ITA only covers establishments meeting industry-and-size thresholds. Establishments under 20 employees are largely absent. Treat absence of an ITA record as "not in scope," not "no injuries."

**Calibration matters more than AUC.** A risk score that ranks well but predicts 50% when reality is 8% is useless for pricing or triage. Calibration plots by decile are produced for every model.

## Pricing model (planned)

| Tier | Price | Audience | Limits |
|---|---|---|---|
| Free | $0 | Researchers, journalists, public | Single-establishment lookup |
| Pro | $79/mo | Independent brokers, consultants | 500 establishments/mo, PDF reports |
| Team | $499/mo | Small carriers, broker offices | Unlimited, API, 5 seats |
| Enterprise | Custom | Regional carriers, large brokers | Data feed, SLA, white-label |

## Roadmap

- [x] Project scaffolding
- [x] Download + load pipeline with schema drift handling
- [x] Establishment-year panel with lags and targets
- [x] Synthetic fixture + end-to-end tests
- [x] Baseline LightGBM model with temporal holdout and isotonic calibration
- [x] Per-establishment risk report (JSON-shaped, ready for API/PDF)
- [x] Enforcement data integration (DOL inspections, citations, penalties)
- [x] Splink-based entity resolution for user-uploaded CSVs
- [x] FastAPI endpoints (POST /score, POST /score-batch)
- [ ] Real-data sanity notebook against 2023+2024 OSHA data
- [ ] PDF report generation
- [ ] Public free-tier search interface
- [ ] Stripe billing for Pro / Team tiers

## License

Code is MIT — see `LICENSE`. The output (risk scores, reports) needs its own terms; do not republish the OSHA-attributed identity of specific high-risk establishments without independent legal review.
