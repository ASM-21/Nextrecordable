"""Generate synthetic OSHA-shaped CSVs for end-to-end pipeline testing.

The real OSHA 300A schema has drifted across years. This fixture intentionally
recreates that drift so the schema resolver and canonical view are actually
exercised: 2018 uses old column names, 2024 uses the new ones with
establishment_id.

Numbers are not realistic in absolute terms but the SHAPE (long tail of zeros,
big employers have higher absolute counts, year-over-year persistence of risk)
is preserved so the panel and modeling code see something sensible.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# Roughly representative high-volume manufacturing / warehousing NAICS
NAICS_POOL = [
    "311612",  # meat processing
    "332710",  # machine shops
    "336411",  # aircraft mfg
    "332999",  # other fabricated metal
    "493110",  # general warehousing
    "311111",  # dog and cat food mfg
    "326199",  # other plastics
    "332111",  # iron and steel forging
]


def _seeded_employees(rng: random.Random) -> int:
    """Log-normal-ish size distribution."""
    return int(rng.lognormvariate(mu=4.5, sigma=1.2))


def _seeded_state(rng: random.Random) -> str:
    return rng.choice(["IL", "OH", "PA", "TX", "CA", "MI", "IN", "GA", "NC"])


def _seeded_zip(rng: random.Random) -> str:
    return f"{rng.randint(10000, 99999)}"


def _seeded_name(i: int) -> str:
    suffix = ["Manufacturing Inc", "Mfg LLC", "Industries", "Products Co", "Works Inc"][i % 5]
    return f"Synthetic Plant {i:04d} {suffix}"


def _street(i: int) -> str:
    return f"{(i * 7) % 9000 + 100} Industrial Park Drive"


def build_establishments(n: int = 300, seed: int = 42) -> list[dict]:
    """Build a stable set of synthetic establishments with intrinsic traits.

    Each establishment has TWO independent risk traits:
      - risk_factor: drives injury counts (operational hazards, work intensity)
      - compliance_factor: drives both citation rates AND a smaller part of
        future injury rates. Real-world dynamic: plants with management
        problems get cited for hazards that don't yet show up as injuries,
        but the same management problems eventually cause injuries.

    This gives enforcement features INDEPENDENT predictive value on top of
    historical injury counts. On real OSHA data this dynamic is real; in
    pure synthetic data we have to inject it explicitly.
    """
    rng = random.Random(seed)
    estabs: list[dict] = []
    for i in range(n):
        risk_factor = rng.lognormvariate(mu=0.0, sigma=0.5)
        # compliance_factor is only weakly correlated with risk_factor
        # (corr ~0.3 by construction)
        compliance_factor = 0.7 * rng.lognormvariate(mu=0.0, sigma=0.5) + 0.3 * risk_factor
        estabs.append(
            {
                "establishment_id": 100_000 + i,
                "establishment_name": _seeded_name(i),
                "street_address": _street(i),
                "city": "Springfield",
                "state": _seeded_state(rng),
                "zip_code": _seeded_zip(rng),
                "naics_code": rng.choice(NAICS_POOL),
                "base_employees": _seeded_employees(rng),
                "risk_factor": risk_factor,
                "compliance_factor": compliance_factor,
            }
        )
    return estabs


def _draw_incidents(
    employees: int, risk_factor: float, rng: random.Random,
    compliance_factor: float = 1.0,
) -> dict:
    """Draw a year's incident counts conditional on size and intrinsic risk.

    compliance_factor adds a smaller independent component that ALSO drives
    future injury rates. This is what lets enforcement features (which
    correlate with compliance_factor) add lift on top of past injury history.
    """
    hours = max(1, int(employees * rng.uniform(1800, 2100)))
    # Expected recordables: baseline 3 per 100 FTE, scaled by risk_factor
    # AND a smaller compliance term (so enforcement features can pick this up).
    expected = (employees / 100.0) * 3.0 * risk_factor * (0.8 + 0.4 * compliance_factor)
    total = max(0, int(rng.gauss(expected, expected**0.5)))
    n_deaths = 1 if rng.random() < 0.001 * risk_factor else 0
    n_dafw = int(total * 0.35)
    n_djtr = int(total * 0.25)
    n_other = max(0, total - n_dafw - n_djtr - n_deaths)
    return {
        "annual_average_employees": employees,
        "total_hours_worked": hours,
        "total_deaths": n_deaths,
        "total_dafw_cases": n_dafw,
        "total_djtr_cases": n_djtr,
        "total_other_cases": n_other,
        "total_injuries": int(total * 0.85),
        "total_skin_disorders": 0,
        "total_respiratory_conditions": 0,
        "total_poisonings": 0,
        "total_hearing_loss": int(total * 0.05),
        "total_other_illnesses": int(total * 0.10),
    }


def write_year_csv(
    establishments: list[dict],
    year: int,
    out_path: Path,
    *,
    use_legacy_schema: bool,
    seed: int = 0,
) -> int:
    """Write one year of synthetic data to a CSV. Returns row count.

    legacy_schema=True simulates the pre-2023 OSHA columns (e.g. `naics` instead
    of `naics_code`, no `establishment_id`). False uses the current schema.
    """
    rng = random.Random((year * 1000) + seed)
    rows: list[dict] = []
    for est in establishments:
        # Year-over-year employee jitter
        employees = max(1, int(est["base_employees"] * rng.uniform(0.85, 1.15)))
        incidents = _draw_incidents(
            employees, est["risk_factor"], rng,
            compliance_factor=est.get("compliance_factor", 1.0),
        )

        if use_legacy_schema:
            row = {
                "company_name": est["establishment_name"],
                "estab_street": est["street_address"],
                "estab_city": est["city"],
                "estab_state": est["state"],
                "estab_zip": est["zip_code"],
                "naics": est["naics_code"],
                "size": "2",
                **incidents,
                "year_filing_for": year,
            }
        else:
            row = {
                "establishment_id": est["establishment_id"],
                "establishment_name": est["establishment_name"],
                "street_address": est["street_address"],
                "city": est["city"],
                "state": est["state"],
                "zip_code": est["zip_code"],
                "naics_code": est["naics_code"],
                "size": "2",
                **incidents,
                "year_filing_for": year,
            }
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def build_synthetic_dataset(root: Path, *, n_establishments: int = 200) -> dict[int, Path]:
    """Build a multi-year synthetic dataset under `root`.

    2019-2022 use the legacy schema (no establishment_id, `naics`, etc.)
    2023-2024 use the current schema with establishment_id.
    """
    estabs = build_establishments(n_establishments)
    out: dict[int, Path] = {}
    for year in (2019, 2020, 2021, 2022, 2023, 2024):
        csv_path = root / f"ita_summary_{year}.csv"
        write_year_csv(
            estabs, year, csv_path, use_legacy_schema=(year < 2023), seed=year
        )
        out[year] = csv_path
    return out
