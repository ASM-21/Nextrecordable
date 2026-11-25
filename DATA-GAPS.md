# Data gaps: finishing the real run with official files

**Status (2026-11-25).** The model, METRICS.md and the SST report run on real data, but the ITA files are third-party copies, and only five years are loaded. The DOL enforcement files are official (data.dol.gov bulk downloads, snapshot 2026-11-15).

**Why it isn't done yet.** The cloud machines this repo is built on cannot reach osha.gov or data.dol.gov. On 2026-11-25 their outbound proxy refused the connection to both hosts before any request reached OSHA, so the 403s first seen on 2026-11-18 most likely came from that network, not from OSHA's CDN. A home or office connection should work. The download links themselves were checked against OSHA's listing page, https://www.osha.gov/itadata, on 2026-11-25: all nine 2016 to 2024 links in `config.py` are unchanged.

| year | what is loaded now | what to load |
|---|---|---|
| 2016, 2017, 2018 | nothing | official summary file |
| 2019, 2020, 2022 | copy from a public GitHub repo, snapshot date unknown | official summary file |
| 2021 | copy, "submitted thru 8-29-2022" | official summary file |
| 2023 | copy, "through 3-16-2024" (early, so late filers look like non-filers) | official file through 12-31-2024. **This year matters most: SST selects on it.** |
| 2024 | nothing | official summary file through 12-31-2025 |
| 2025 | nothing | published 2026 only as an early CSV "through 08-31-2026"; left out on purpose (see the note under `ITA_SUMMARY_URLS` in `config.py`) |

Copies came from `github.com/nbrady387/DSCI-235-Final` at commit `936d58068d3262ea099e031ea1ac82b5c4c918c2`, a 2024 student project. They match OSHA's column layout, but nothing confirms they are unmodified.

## One command (about an hour, most of it waiting)

On a machine that can reach osha.gov, from the repo root with the virtualenv active and the existing `data/` folder in place:

```bash
nextrecordable refresh-official
```

It stops at the first failed check, and a rerun keeps finished downloads. In order, it:

1. Checks every 2016 to 2024 URL and stops before downloading anything if one is gone. A 403 on every URL means the network is refusing osha.gov; try another connection. A 404 on one means OSHA renamed it: fix it in `nextrecordable/config.py` from the listing page.
2. Downloads the nine summary zips to `data/raw/ita_summary/`.
3. Checks each zip opens, holds a CSV, and has the columns the build needs. A saved error page fails here.
4. Compares each official year with the copy it replaces (rows, distinct establishments, death, case and hours totals) and writes `reports/official_vs_copy.json`. Counts only. 2023 will differ, because the copy stopped in March 2024; any other year that differs is worth a line in FINDINGS.md.
5. Moves the copies to `data/raw/ita_summary/third_party_copies/`. Nothing is deleted.
6. Rewrites `data/raw/SOURCES.json` as official, keeping the DOL entries; the old file stays as `SOURCES.previous.json`. With no third-party file declared, the data notice drops out of every output by itself.
7. Runs `load`, `build` (about 11 minutes on 4 cores: it links 1.8 million inspections), `sanity`, `train` (about a minute) and `falsify` (about 2 minutes).

Add `--refresh-dol` to download fresh enforcement files from data.dol.gov as well (adds the DOL downloads and a `load-enforcement` step), or `--no-train` to stop after step 6.

Then:

1. With 2016 to 2024 loaded the test year moves to 2023 (predicting 2024), so the headline numbers change. `pytest tests/test_docs.py` fails until FINDINGS.md and the README's results quote the new `reports/*.json` values; edit the numbers it names, and add a line on anything `official_vs_copy.json` shows.
2. Update the README status line and the FINDINGS.md opening paragraph, which still say the files are copies.
3. Optional: `nextrecordable synthetic-baseline` refreshes the comparison run in METRICS.md (about a minute).
4. Commit `reports/` and the edited docs. Never commit `data/` (it is gitignored).

## The same steps by hand

`nextrecordable download summary`, move `data/raw/ita_summary/ita_summary_20*.csv` aside, write `data/raw/SOURCES.json` with `"kind": "official"` and one entry per downloaded zip (format in `provenance.py`; the build hashes each file), then `nextrecordable load`, `build`, `sanity`, `train` and `falsify`.

## Enforcement files, if they need refreshing

```bash
curl -O https://data.dol.gov/data-catalog/OSHA/inspection/OSHA_inspection.zip
```

```bash
curl -O https://data.dol.gov/data-catalog/OSHA/violation/OSHA_violation.zip
```

Or run `nextrecordable refresh-official --refresh-dol`, which does all of this. By hand: unzip each into `data/raw/dol/inspection/` and `data/raw/dol/violation/` (one folder of `*_chunk_<n>.csv` files each), then:

```bash
nextrecordable load-enforcement --inspection data/raw/dol/inspection --violation data/raw/dol/violation
```

## Not used, on purpose

- **Form 300/301 case detail** (`ITA_CASE_DETAIL_URLS`). Not loaded and not needed for anything in the product. It holds incident narratives about individual workers; if it is ever loaded for injury-mix features, no field from it may reach a customer-facing report (`tests/test_sst.py::test_report_uses_only_summary_data`).
- **DOL `osha_accident`** was downloaded but is not used yet.
