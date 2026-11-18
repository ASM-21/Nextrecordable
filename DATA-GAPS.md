# Data gaps: finishing the real run with official files

**Status (2026-11-18).** The model, METRICS.md and the SST report run on real data, but the ITA files are third-party copies, and only five years are loaded. osha.gov's CDN answered 403 to every request from the build machine, robots.txt included. The DOL enforcement files are official (data.dol.gov bulk downloads, snapshot 2026-11-15) and need nothing.

| year | what is loaded now | what to load |
|---|---|---|
| 2016, 2017, 2018 | nothing | official summary file |
| 2019, 2020, 2022 | copy from a public GitHub repo, snapshot date unknown | official summary file |
| 2021 | copy, "submitted thru 8-29-2022" | official summary file |
| 2023 | copy, "through 3-16-2024" (early, so late filers look like non-filers) | official file through 12-31-2024. **This year matters most: SST selects on it.** |
| 2024 | nothing | official summary file through 12-31-2025 |
| 2025 | nothing | if OSHA has published it, add its URL to `config.py` |

Copies came from `github.com/nbrady387/DSCI-235-Final` at commit `936d58068d3262ea099e031ea1ac82b5c4c918c2`, a 2024 student project. They match OSHA's column layout, but nothing confirms they are unmodified.

## Steps (about 45 minutes, most of it waiting)

Run on a machine that can reach osha.gov, from the repo root with the virtualenv active.

1. Check the URLs still resolve. OSHA renames files now and then; the listing page is https://www.osha.gov/Establishment-Specific-Injury-and-Illness-Data.

```bash
python scripts/check_source_urls.py
```

2. Fix any URL that failed in `nextrecordable/config.py` (`ITA_SUMMARY_URLS`), using the link text on the listing page. These are the names as of 2026-05-15:

| year | URL |
|---|---|
| 2016 | https://www.osha.gov/sites/default/files/ITA%20Data%20CY%202016.zip |
| 2017 | https://www.osha.gov/sites/default/files/ITA%20Data%20CY%202017.zip |
| 2018 | https://www.osha.gov/sites/default/files/ITA%20Data%20CY%202018.zip |
| 2019 | https://www.osha.gov/sites/default/files/ITA%20Data%20CY%202019.zip |
| 2020 | https://www.osha.gov/sites/default/files/ITA-Data-CY-2020.zip |
| 2021 | https://www.osha.gov/sites/default/files/ITA-data-cy2021.zip |
| 2022 | https://www.osha.gov/sites/default/files/ITA-data-cy2022.zip |
| 2023 | https://www.osha.gov/sites/default/files/ITA_300A_Summary_Data_2023_through_12-31-2024.zip |
| 2024 | https://www.osha.gov/sites/default/files/ITA_300A_Summary_Data_2024_through_12-31-2025.zip |

3. Download. Files land in `data/raw/ita_summary/ita_summary_<year>.zip`. When a year has both a zip and a CSV copy, the loader uses the zip, but remove the copies anyway so nothing ambiguous remains:

```bash
nextrecordable download summary
```

```bash
rm data/raw/ita_summary/ita_summary_20*.csv
```

4. Optional, ten minutes: check whether the copies were faithful. Before deleting them in step 3, compare row counts and a checksum of the case columns per year; a mismatch is worth a line in FINDINGS.md.

5. Declare the sources. Replace `data/raw/SOURCES.json` with this, adding one line per year you downloaded (the build hashes each file):

```json
{
  "kind": "official",
  "note": "ITA files downloaded from osha.gov; DOL files from data.dol.gov.",
  "files": {
    "ita_summary/ita_summary_2023.zip": {"origin": "https://www.osha.gov/sites/default/files/ITA_300A_Summary_Data_2023_through_12-31-2024.zip", "snapshot": "through 2024-12-31"},
    "dol/OSHA_inspection.zip": {"origin": "https://data.dol.gov/data-catalog/OSHA/inspection/OSHA_inspection.zip", "snapshot": "2026-11-15"},
    "dol/OSHA_violation.zip": {"origin": "https://data.dol.gov/data-catalog/OSHA/violation/OSHA_violation.zip", "snapshot": "2026-11-15"}
  }
}
```

Once no file is declared `third_party_copy`, the data notice disappears from every output by itself.

6. Rebuild and retrain. The full build links 1.8 million inspections and takes about 11 minutes on 4 cores; training takes about a minute; the falsification pass about 2 minutes; the synthetic baseline about 1 minute.

```bash
nextrecordable load
```

```bash
nextrecordable build
```

```bash
nextrecordable sanity && nextrecordable train && nextrecordable falsify
```

7. Update the prose. With 2016 to 2024 loaded the test year moves to 2023 (predicting 2024), so the headline numbers change. `pytest tests/test_docs.py` fails until FINDINGS.md and the README's results quote the new `reports/*.json` values; edit the numbers it names.

8. Commit `reports/` and the edited docs. Never commit `data/` (it is gitignored).

## Enforcement files, if they need refreshing

```bash
curl -O https://data.dol.gov/data-catalog/OSHA/inspection/OSHA_inspection.zip
```

```bash
curl -O https://data.dol.gov/data-catalog/OSHA/violation/OSHA_violation.zip
```

Unzip each into `data/raw/dol/inspection/` and `data/raw/dol/violation/` (one folder of `*_chunk_<n>.csv` files each), then:

```bash
nextrecordable load-enforcement --inspection data/raw/dol/inspection --violation data/raw/dol/violation
```

## Not used, on purpose

- **Form 300/301 case detail** (`ITA_CASE_DETAIL_URLS`). Not loaded and not needed for anything in the product. It holds incident narratives about individual workers; if it is ever loaded for injury-mix features, no field from it may reach a customer-facing report (`tests/test_sst.py::test_report_uses_only_summary_data`).
- **DOL `osha_accident`** was downloaded but is not used yet.
