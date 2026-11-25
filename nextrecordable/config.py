"""Configuration: data source URLs, paths, schema hints.

URLs were verified against OSHA's listing page, https://www.osha.gov/itadata
(formerly /Establishment-Specific-Injury-and-Illness-Data), on 2026-11-25; all
2016-2024 links were unchanged since 2026-05-15. OSHA occasionally shifts file
naming conventions. If a download returns 404, re-fetch the listing page and
update these URLs.
"""

import os
from pathlib import Path

# ---------- Filesystem ----------
# Paths default to the current working directory, so an installed package
# works from any checkout. Override with environment variables.

DATA_DIR = Path(os.environ.get("NEXTRECORDABLE_DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = Path(os.environ.get("NEXTRECORDABLE_DB_PATH", str(DATA_DIR / "nextrecordable.duckdb")))
MODEL_DIR = Path(os.environ.get("NEXTRECORDABLE_MODEL_DIR", str(DATA_DIR / "model")))
REPORTS_DIR = Path(os.environ.get("NEXTRECORDABLE_REPORTS_DIR", "reports"))

# ---------- ITA Summary (Form 300A) ----------
# Coverage: all establishments meeting reporting thresholds. ~hundreds of thousands of rows per year.
ITA_SUMMARY_URLS: dict[int, str] = {
    2016: "https://www.osha.gov/sites/default/files/ITA%20Data%20CY%202016.zip",
    2017: "https://www.osha.gov/sites/default/files/ITA%20Data%20CY%202017.zip",
    2018: "https://www.osha.gov/sites/default/files/ITA%20Data%20CY%202018.zip",
    2019: "https://www.osha.gov/sites/default/files/ITA%20Data%20CY%202019.zip",
    2020: "https://www.osha.gov/sites/default/files/ITA-Data-CY-2020.zip",
    2021: "https://www.osha.gov/sites/default/files/ITA-data-cy2021.zip",
    2022: "https://www.osha.gov/sites/default/files/ITA-data-cy2022.zip",
    2023: "https://www.osha.gov/sites/default/files/ITA_300A_Summary_Data_2023_through_12-31-2024.zip",
    2024: "https://www.osha.gov/sites/default/files/ITA_300A_Summary_Data_2024_through_12-31-2025.zip",
}
# Left out on purpose: 2025 is published (as of 2026-11-25) only as an early
# snapshot, a bare CSV "through 08-31-2026" at
# https://www.osha.gov/sites/default/largefiles/ITA_300A_Summary_Data_2025_through_08-31-2026.csv
# Adding it would make 2025 the outcome year, and late filers in an early
# snapshot look like non-filers (the problem the old 2023 copy had). The
# downloader also expects zips. Add it once OSHA posts the through-12-31 file.

# ---------- ITA Case Detail (Forms 300/301) ----------
# Coverage: establishments with 100+ employees in designated high-hazard industries.
# Includes case-level narratives starting 2023. OIICS coded as of recent updates.
ITA_CASE_DETAIL_URLS: dict[int, str] = {
    2023: "https://www.osha.gov/sites/default/largefiles/ITA_Case_Detail_Data_2023_through_12-31-2023OIICS.zip",
    2024: "https://www.osha.gov/sites/default/files/ITA_Case_Detail_Data_2024_through_12-31-2025.zip",
}

# ---------- DOL enforcement data ----------
# Catalog page; specific files discovered at runtime.
DOL_ENFORCEMENT_CATALOG = "https://enforcedata.dol.gov/views/data_catalogs.php"

# ---------- Schema notes ----------
# Column names drift across years. Known synonyms we resolve at load time:
NAICS_COLUMN_CANDIDATES = ("naics_code", "naics", "industry_code")
NAME_COLUMN_CANDIDATES = ("establishment_name", "company_name", "estab_name")
STREET_COLUMN_CANDIDATES = ("street_address", "street", "estab_street")
CITY_COLUMN_CANDIDATES = ("city", "estab_city")
STATE_COLUMN_CANDIDATES = ("state", "estab_state")
ZIP_COLUMN_CANDIDATES = ("zip_code", "zip", "estab_zip")
SIZE_COLUMN_CANDIDATES = ("size", "size_code", "establishment_size")
ANNUAL_EMPLOYEES_CANDIDATES = ("annual_average_employees", "annual_avg_employees")
HOURS_CANDIDATES = ("total_hours_worked", "total_hrs_worked")
