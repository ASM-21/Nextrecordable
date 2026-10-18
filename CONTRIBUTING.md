# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q && ruff check .
```

Tests use synthetic files only (`tests/synthetic.py`, `tests/synthetic_enforcement.py`); no network, no real data.

## Conventions

- **One implementation of every normalization rule.** Names, addresses, ZIPs and states are normalized in `normalize.py` and nowhere else. Pipeline steps map distinct raw values through those functions and join the results back (`keys.py`). If SQL needs a normalized value, it joins it in; it never re-implements the rule. Two copies of "the same" rule drifted apart once, and the soft-key joins silently returned nothing.
- **Text in, types explicit.** Raw CSVs load with every column as VARCHAR. Type inference turned ZIPs into integers once and dropped their leading zeros.
- **Pipeline steps are functions of a DuckDB connection** that build one table and return its row count. They compose, and tests call them directly.
- **Per-request work on a per-request cursor.** API handlers take `con.cursor()` and keep scratch rows in TEMP tables, which are private to a cursor.
- **Synthetic fixtures reproduce the real files' traps** (schema drift, size-code split, leading-zero ZIPs, amended filings, skipped years, DOL letter codes and integer ZIPs). When real data surprises you, teach the generator the surprise first.
- **A bug gets a reproduction before a fix, and the reproduction becomes a test.** The README lists every one so far.
- **No pickles.** Model artifacts are LightGBM text files plus JSON.

## Changing the matcher

Weights live at the top of `match.py`. Any change must keep `tests/test_match_quality.py` green: per-case accuracy floors, zero wrong matches, and the hard negatives (changed plant numbers, similar names in one ZIP). If you tune for recall, add the hard negative that would have caught the regression.

## Adding a model feature

1. Produce the raw column in a loader or `enforcement.py`.
2. Add it to the panel (`features.py`) or the enforced panel. Use only information available when the forecast is made: a panel row for year t forecasts year t+1 and is scored after year-t ITA data is published.
3. Add it to `NUMERIC_FEATURES`, `ENFORCEMENT_FEATURES` or `CATEGORICAL_FEATURES` and give it a label in `FEATURE_LABELS`.
4. Teach the synthetic generator to carry real signal through it, and add a test that it is used.
5. Read `reports/METRICS.md` against the baselines before keeping it.

## Adding a data gate

Gates live in `sanity.py`. Each records the measured value, the rule, and pass / warn / fail. A fail stops `run-all`; prefer warn unless a failure would make the model meaningless.

## Good first contributions

- Run `nextrecordable run-all` on real data and open an issue with your `reports/METRICS.md`. This is the most useful thing anyone can do right now.
- Add features from DOL `osha_accident` (hospitalization and amputation reports).
- Use the ITA case-detail files (Forms 300/301, 2023 onward) for injury-type mix.
- Add BLS SOII industry rates as a second benchmark next to the NAICS-4 pooled rate.
- Serve from read-only DuckDB copies so several processes can share one trained database.

## Before opening a PR

```bash
ruff check . && pytest -q
```

CI runs both on Python 3.11 to 3.13 for every PR, and weekly against the newest allowed dependency versions.
