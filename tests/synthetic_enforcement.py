"""Synthetic DOL enforcement data linked to the ITA establishments.

Produces inspection and violation CSVs shaped like the real DOL data, where
each inspection's estab_name + site_state + site_zip match an establishment
in the ITA fixture (so the soft_key join works).

Design choices baked in for realism:
  - High-risk establishments (high risk_factor) are MORE LIKELY to be
    inspected, and inspections MORE OFTEN result in citations.
  - Each establishment also has a SEPARATE "enforcement-prone" trait
    (compliance_factor) that's only weakly correlated with risk_factor.
    This captures the real-world dynamic where citations encode hazards
    that injury counts alone don't reveal (a plant can have low recent
    recordables but still get cited for serious safety violations during
    a complaint-driven inspection, and those citations predict future
    incidents). This is why enforcement features add lift in production.
  - Penalties scale with severity.
  - Inspection types mix programmed (planned) and unprogrammed
    (complaint, accident, referral) following OSHA's actual proportions.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .synthetic import build_establishments

INSPECTION_TYPES = [
    ("Programmed Planned", 0.35),
    ("Unprog Complaint", 0.25),
    ("Unprog Accident", 0.10),
    ("Unprog Referral", 0.10),
    ("Programmed Related", 0.15),
    ("Unprog Fatal/Catastrophe", 0.05),
]
INSPECTION_TYPE_NAMES = [t[0] for t in INSPECTION_TYPES]
INSPECTION_TYPE_WEIGHTS = [t[1] for t in INSPECTION_TYPES]

VIOLATION_SEVERITIES = [
    ("Other", 0.45),
    ("Serious", 0.45),
    ("Repeat", 0.07),
    ("Willful", 0.03),
]
VIOL_SEV_NAMES = [v[0] for v in VIOLATION_SEVERITIES]
VIOL_SEV_WEIGHTS = [v[1] for v in VIOLATION_SEVERITIES]

PENALTY_BY_SEVERITY = {
    "Other": (0, 2000),
    "Serious": (1000, 15000),
    "Repeat": (5000, 70000),
    "Willful": (10000, 156000),
}

STANDARDS_POOL = [
    "19100132", "19100147", "19100212", "19100147(c)(2)(i)",
    "19100178", "19100305(b)(1)", "19101200", "19100023(c)(1)",
    "19100147(c)(4)(ii)", "19260501(b)(1)",
]


def _probability_inspected(risk_factor: float, year: int, rng: random.Random) -> float:
    """Probability this establishment is inspected this year.

    Synthetic rate is INTENTIONALLY HIGHER than real OSHA (~4% per year).
    Real data needs no such inflation because the volume is vastly larger
    (millions of establishment-years) so even 4% gives plenty of signal.
    Our synthetic fixture is small (a few thousand rows) so we boost the
    rate to ~20-30% to give tests detectable signal.
    """
    base = 0.20
    year_factor = 1.0 + (year - 2019) * 0.02
    p = base * (0.5 + risk_factor) * year_factor
    return min(p, 0.6)


def _draw_n_citations(insp_type: str, compliance_factor: float, rng: random.Random) -> int:
    """How many citations come out of this inspection.

    Driven by compliance_factor (independent of risk_factor in the establishment
    generator). This is what gives enforcement features predictive power that
    isn't already in injury counts.
    """
    if "Programmed" in insp_type:
        base = 0.8 * compliance_factor
    elif "Fatal" in insp_type:
        base = 4.0 * compliance_factor
    else:
        base = 1.6 * compliance_factor

    n = max(0, int(rng.gauss(base, base**0.5)))
    return n


def build_synthetic_enforcement(
    out_dir: Path,
    *,
    n_establishments: int = 200,
    years: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024),
    seed: int = 7,
) -> tuple[Path, Path]:
    """Generate inspection and violation CSVs.

    Returns (inspection_csv_path, violation_csv_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    estabs = build_establishments(n=n_establishments, seed=42)
    rng = random.Random(seed)

    inspections: list[dict] = []
    violations: list[dict] = []
    activity_nr = 1_000_000

    for year in years:
        for est in estabs:
            if rng.random() >= _probability_inspected(est["risk_factor"], year, rng):
                continue

            activity_nr += 1
            insp_type = rng.choices(INSPECTION_TYPE_NAMES, weights=INSPECTION_TYPE_WEIGHTS, k=1)[0]
            open_month = rng.randint(1, 12)
            open_day = rng.randint(1, 28)
            inspections.append(
                {
                    "activity_nr": activity_nr,
                    "estab_name": est["establishment_name"],
                    "site_address": est["street_address"],
                    "site_city": est["city"],
                    "site_state": est["state"],
                    "site_zip": est["zip_code"],
                    "naics_code": est["naics_code"],
                    "insp_type": insp_type,
                    "insp_scope": "Complete" if rng.random() > 0.3 else "Partial",
                    "open_date": f"{year}-{open_month:02d}-{open_day:02d}",
                    "close_case_date": f"{year}-{min(12, open_month + 3):02d}-{open_day:02d}",
                }
            )

            n_cit = _draw_n_citations(insp_type, est["compliance_factor"], rng)
            for c in range(n_cit):
                sev = rng.choices(VIOL_SEV_NAMES, weights=VIOL_SEV_WEIGHTS, k=1)[0]
                lo, hi = PENALTY_BY_SEVERITY[sev]
                initial = rng.randint(lo, hi)
                # Final penalty often reduced via informal settlement
                current = int(initial * rng.uniform(0.5, 1.0))
                violations.append(
                    {
                        "activity_nr": activity_nr,
                        "citation_id": f"{activity_nr}-{c + 1}",
                        "standard": rng.choice(STANDARDS_POOL),
                        "viol_type": sev,
                        "gravity": rng.randint(1, 10),
                        "initial_penalty": initial,
                        "current_penalty": current,
                    }
                )

    insp_path = out_dir / "osha_inspection.csv"
    viol_path = out_dir / "osha_violation.csv"

    if inspections:
        with open(insp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(inspections[0].keys()))
            w.writeheader()
            w.writerows(inspections)
    if violations:
        with open(viol_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(violations[0].keys()))
            w.writeheader()
            w.writerows(violations)

    return insp_path, viol_path
