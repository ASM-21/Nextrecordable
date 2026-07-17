# Contributing / working on this repo

Notes for future-you (or anyone else) picking this project back up.

## Setup

```bash
git clone <repo-url>
cd osha-risk
python -m venv .venv && source .venv/bin/activate
make dev-install
make test
```

If `make test` passes on a clean clone, the environment is good.

## Repo conventions

- **DuckDB is the only datastore.** No Postgres, no external services for
  local dev. Everything lives in `data/osha.duckdb` (gitignored).
- **Every pipeline stage is a pure function over a DuckDB connection.**
  `build_X(con, ...) -> int` (row count) is the standard shape. Makes stages
  composable and testable without touching disk.
- **Synthetic fixtures over mocks.** `tests/synthetic.py` and
  `tests/synthetic_enforcement.py` generate realistic OSHA-shaped data with
  intentional schema drift and independent risk signals baked in. Prefer
  extending these over mocking DuckDB calls.
- **Schema drift is real and expected.** OSHA/DOL column names change across
  years. `schema.py` and `enforcement.py`'s column-mapping dicts are the
  single source of truth for known synonyms. When real data introduces a new
  column name, add it to the relevant `*_CANDIDATES` tuple rather than
  special-casing it downstream.
- **Soft keys vs hard keys.** `estab_key` (md5 of normalized name+state+zip5)
  is the PRIMARY identifier used everywhere, even though `establishment_id`
  exists in recent ITA years. This is deliberate — see the docstring on
  `features.build_establishment_key_view` for why using the hard ID would
  break lag features at the 2022/2023 schema boundary.

## Before opening a PR

```bash
make fix    # auto-fix lint issues
make lint   # confirm clean
make test   # confirm 100% pass
```

CI runs both on every push/PR (`.github/workflows/ci.yml`). It also runs a
non-blocking weekly job that pings OSHA's data URLs to catch link rot —
check that job if downloads start failing and nothing in this repo changed.

## Adding a new feature to the model

1. Add the raw column(s) via a loader (`load.py` or `enforcement.py` style)
2. Wire it into the panel (`features.py` or `enforcement.py`'s
   `attach_*_to_panel` pattern — always LAG features that wouldn't be known
   at prediction time)
3. Add the feature name to `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` in
   `models/baseline.py`
4. Extend the synthetic fixture so the new feature has realistic signal
   (see how `compliance_factor` was added to `tests/synthetic.py` and
   `tests/synthetic_enforcement.py` to give enforcement features real
   predictive value independent of injury history)
5. Add a test asserting the new feature appears with nonzero importance,
   following `test_enforcement.py::test_model_uses_enforcement_features`

## Known rough edges

- Splink hits internal errors ("Salting partitions must be > 1") on very
  small datasets. `match.py` wraps training calls in
  `contextlib.suppress(Exception)` and falls back to default parameter
  values, which is fine for our threshold-based use but worth revisiting if
  match quality looks off on real data.
- The soft-key hash in `match.py`'s `fallback_softkey_match` and in
  `features.py`'s `build_establishment_key_view` MUST use identical SQL
  normalization (not the Python `normalize_name`, which strips entity
  suffixes and would produce a different hash). If you touch either, check
  both.
- Real DOL enforcement URLs are unstable (the agency is mid-migration to a
  new open data portal as of 2026). `enforcement.py`'s loaders accept local
  file paths rather than hardcoding URLs for this reason — download manually
  from whatever the current portal is and point the loader at the file.
