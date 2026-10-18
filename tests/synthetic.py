"""Synthetic ITA 300A files shaped like the real ones, including their traps.

What the generator reproduces on purpose, because each one broke the
pipeline at some point:

  * schema drift: 2019-2022 files use legacy column names (company_name,
    naics, estab_zip); 2023+ use the current names plus establishment_id,
    created_timestamp and year_filing_for;
  * size code 2 split into 21/22 starting with 2023 data;
  * leading-zero ZIPs (CT, MA, NJ), some written as ZIP+4;
  * amended filings: a second row for the same establishment and year with a
    later created_timestamp;
  * skipped years: some establishments don't file one year.

Each establishment has two latent traits: risk_factor (drives injuries) and
compliance_factor (drives citations and part of injuries), so enforcement
data carries signal that injury history alone doesn't.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

YEARS = (2019, 2020, 2021, 2022, 2023, 2024)

NAICS_POOL = ["311612", "332710", "336411", "332999", "493110", "311111", "326199", "332111"]

STATE_ZIPS: dict[str, tuple[int, int]] = {
    "IL": (60001, 62999), "OH": (43001, 45999), "PA": (15001, 19699), "TX": (75001, 79999),
    "CA": (90001, 96199), "MI": (48001, 49999), "IN": (46001, 47999), "GA": (30001, 31999),
    "CT": (6001, 6999), "MA": (1001, 2799), "NJ": (7001, 8999),
}

ADJ = ["Acme", "Summit", "Keystone", "Lakeside", "Riverbend", "Granite", "Prairie", "Allied",
       "Precision", "Midwest", "Northstar", "Harbor", "Pioneer", "Liberty", "Heartland", "Ironwood",
       "Bluewater", "Cardinal", "Evergreen", "Frontier", "Gateway", "Highland", "Junction", "Lincoln"]
NOUN = ["Forge", "Castings", "Plastics", "Foods", "Logistics", "Machining", "Fabrication",
        "Packaging", "Distribution", "Tool and Die", "Metal Works", "Polymers", "Assembly",
        "Stamping", "Extrusion", "Coatings"]
SUFFIX = ["Inc", "LLC", "Co", "Corporation", "Manufacturing Inc", "Industries", ""]
STREETS = ["Industrial Park Dr", "Commerce Way", "Factory Rd", "Mill St", "Enterprise Blvd",
           "Rail Spur Rd", "Harbor Ave", "Quarry Ln"]


def build_establishments(n: int = 300, seed: int = 42) -> list[dict]:
    """Stable synthetic establishments with latent risk traits."""
    rng = random.Random(seed)
    estabs: list[dict] = []
    used: set[tuple[str, str, str]] = set()
    states = list(STATE_ZIPS)
    for i in range(n):
        while True:
            name = f"{rng.choice(ADJ)} {rng.choice(NOUN)}"
            if rng.random() < 0.15:
                name += f" Plant {rng.randint(1, 30)}"
            suffix = rng.choice(SUFFIX)
            name = f"{name} {suffix}".strip()
            state = rng.choice(states)
            lo, hi = STATE_ZIPS[state]
            z = f"{rng.randint(lo, hi):05d}"
            if (name, state, z) not in used:
                used.add((name, state, z))
                break
        risk_factor = rng.lognormvariate(0.0, 0.5)
        compliance_factor = 0.7 * rng.lognormvariate(0.0, 0.5) + 0.3 * risk_factor
        estabs.append({
            "establishment_id": 100_000 + i,
            "establishment_name": name,
            "street_address": f"{(i * 37) % 9000 + 100} {rng.choice(STREETS)}",
            "city": f"Town{i % 50}",
            "state": state,
            "zip_code": z,
            "naics_code": rng.choice(NAICS_POOL),
            "base_employees": max(5, int(rng.lognormvariate(4.5, 1.1))),
            "risk_factor": risk_factor,
            "compliance_factor": compliance_factor,
            "skip_year": rng.choice(YEARS[1:-1]) if rng.random() < 0.08 else None,
            "amended_year": rng.choice(YEARS) if rng.random() < 0.05 else None,
        })
    return estabs


def _size_code(employees: int, year: int) -> str:
    if employees < 20:
        return "1"
    if employees < 250:
        if year >= 2023:
            return "21" if employees < 100 else "22"
        return "2"
    return "3"


def _incidents(employees: int, risk: float, compliance: float, rng: random.Random) -> dict:
    hours = max(1, int(employees * rng.uniform(1800, 2100)))
    expected = (employees / 100.0) * 3.0 * risk * (0.8 + 0.4 * compliance)
    total = max(0, round(rng.gauss(expected, max(expected, 0.1) ** 0.5)))
    deaths = 1 if rng.random() < 0.001 * risk else 0
    dafw = int(total * 0.35)
    djtr = int(total * 0.25)
    other = max(0, total - dafw - djtr - deaths)
    return {
        "annual_average_employees": employees,
        "total_hours_worked": hours,
        "total_deaths": deaths,
        "total_dafw_cases": dafw,
        "total_djtr_cases": djtr,
        "total_other_cases": other,
        "total_injuries": int(total * 0.85),
        "total_hearing_loss": int(total * 0.05),
        "total_other_illnesses": int(total * 0.10),
    }


def write_year_csv(establishments: list[dict], year: int, out_path: Path, *,
                   use_legacy_schema: bool, seed: int = 0) -> int:
    """One year's 300A file. Returns rows written."""
    rng = random.Random(year * 1000 + seed)
    rows: list[dict] = []
    for est in establishments:
        employees = max(1, int(est["base_employees"] * rng.uniform(0.85, 1.15)))
        inc = _incidents(employees, est["risk_factor"], est["compliance_factor"], rng)
        if est["skip_year"] == year:
            continue
        zip_value = est["zip_code"] if rng.random() > 0.2 else f"{est['zip_code']}-{rng.randint(1000, 9999)}"
        if use_legacy_schema:
            row = {
                "company_name": est["establishment_name"],
                "estab_street": est["street_address"],
                "estab_city": est["city"],
                "estab_state": est["state"],
                "estab_zip": zip_value,
                "naics": est["naics_code"],
                "size": _size_code(employees, year),
                **inc,
                "year_filing_for": year,
            }
        else:
            row = {
                "id": f"{year}{est['establishment_id']}",
                "establishment_name": est["establishment_name"],
                "establishment_id": est["establishment_id"],
                "company_name": "Synthetic Holdings",
                "street_address": est["street_address"],
                "city": est["city"],
                "state": est["state"],
                "zip_code": zip_value,
                "naics_code": est["naics_code"],
                "size": _size_code(employees, year),
                **inc,
                "created_timestamp": f"{year + 1}-02-{rng.randint(1, 28):02d} 10:00:00",
                "year_filing_for": year,
            }
        rows.append(row)
        if est["amended_year"] == year:
            amended = dict(row)
            amended["total_other_cases"] = row["total_other_cases"] + 1
            if not use_legacy_schema:
                amended["created_timestamp"] = f"{year + 1}-06-15 09:00:00"
            rows.append(amended)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def build_synthetic_dataset(root: Path, *, n_establishments: int = 200, seed: int = 42) -> dict[int, Path]:
    """Six years of 300A files under `root`: legacy schema through 2022, current after."""
    estabs = build_establishments(n_establishments, seed=seed)
    out: dict[int, Path] = {}
    for year in YEARS:
        path = Path(root) / f"ita_summary_{year}.csv"
        write_year_csv(estabs, year, path, use_legacy_schema=year < 2023, seed=year)
        out[year] = path
    return out
