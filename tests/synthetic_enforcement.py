"""Synthetic DOL osha_inspection / osha_violation files in the real format.

Matches what the real files do, per a mirror of the DOL data and IRE's code
book: single-letter insp_type and viol_type codes, delete_flag 'X' for
deleted citations, site_zip stored as an integer (leading zeros lost), and
employer names spelled differently from their ITA filings. A share of
inspections hit establishments that never file with ITA, as most real
inspections do; those must stay unlinked.

Since the first real run it also copies the data.dol.gov bulk layout:
upper-case headers, dates as "2019-03-05 00:00:00+00:00", and pandas'
placeholders for missing values ("nan" penalties, "NaT" dates).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .synthetic import YEARS, build_establishments

INSP_TYPES = [("H", 0.35), ("B", 0.25), ("C", 0.10), ("I", 0.15), ("F", 0.10), ("A", 0.05)]
VIOL_TYPES = [("O", 0.45), ("S", 0.45), ("R", 0.07), ("W", 0.03)]
PENALTY = {"O": (0, 2000), "S": (1000, 15000), "R": (5000, 70000), "W": (10000, 156000), "U": (0, 1000)}
STANDARDS = ["19100212", "19100147", "19100178", "19100305", "19101200", "19100023", "19260501"]


def dol_spelling(name: str, rng: random.Random) -> str:
    """How DOL tends to spell an employer that files with ITA under `name`."""
    r = rng.random()
    if r < 0.45:
        return name
    if r < 0.65:
        return name.upper()
    if r < 0.80:
        return name.replace("Manufacturing", "MFG").replace("Corporation", "Corp.").replace(" and ", " & ")
    if r < 0.92:
        for suffix in (" Inc", " LLC", " Co", " Corporation", " Industries"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name + ", Inc."
    words = name.split()
    i = rng.randrange(len(words))
    w = words[i]
    if len(w) > 4:
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _probability_inspected(compliance: float, year: int) -> float:
    # Far above OSHA's real ~4% a year so a small fixture has signal to learn from.
    return min(0.6, 0.20 * (0.5 + compliance) * (1.0 + (year - YEARS[0]) * 0.02))


def build_synthetic_enforcement(
    out_dir: Path,
    *,
    n_establishments: int = 200,
    n_non_filers: int | None = None,
    seed: int = 7,
    estab_seed: int = 42,
) -> tuple[Path, Path]:
    """Write osha_inspection.csv and osha_violation.csv. Returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    estabs = build_establishments(n_establishments, seed=estab_seed)
    non_filers = build_establishments(n_non_filers if n_non_filers is not None else n_establishments // 2,
                                      seed=estab_seed + 1000)
    for nf in non_filers:
        nf["establishment_name"] = "Smallshop " + nf["establishment_name"]
    inspections, violations = [], []
    activity_nr = 300_000_000
    for year in YEARS:
        for est in estabs + non_filers:
            if rng.random() >= _probability_inspected(est["compliance_factor"], year):
                continue
            activity_nr += 1
            itype = rng.choices([t for t, _ in INSP_TYPES], weights=[w for _, w in INSP_TYPES])[0]
            month = rng.randint(1, 12)
            inspections.append({
                "activity_nr": activity_nr,
                "estab_name": dol_spelling(est["establishment_name"], rng),
                "site_address": est["street_address"].upper(),
                "site_city": est["city"].upper(),
                "site_state": est["state"],
                "site_zip": int(est["zip_code"]),  # DOL stores ZIP as an integer
                "naics_code": est["naics_code"],
                "insp_type": itype,
                "insp_scope": rng.choice(["A", "B"]),
                "open_date": f"{year}-{month:02d}-{rng.randint(1, 28):02d} 00:00:00+00:00",
                "close_case_date": "NaT",
            })
            base = {"H": 0.8, "I": 0.8, "A": 4.0}.get(itype, 1.6) * est["compliance_factor"]
            n_cit = max(0, round(rng.gauss(base, base ** 0.5)))
            for c in range(n_cit):
                vtype = rng.choices([t for t, _ in VIOL_TYPES], weights=[w for _, w in VIOL_TYPES])[0]
                lo, hi = PENALTY[vtype]
                initial = rng.randint(lo, hi)
                violations.append({
                    "activity_nr": activity_nr,
                    "citation_id": f"01{c + 1:03d}",
                    "delete_flag": "X" if rng.random() < 0.05 else "",
                    "standard": rng.choice(STANDARDS),
                    "viol_type": vtype,
                    "issuance_date": f"{year}-{min(12, month + 2):02d}-15 00:00:00+00:00",
                    "initial_penalty": "nan" if rng.random() < 0.05 else f"{initial}.00",
                    "current_penalty": int(initial * rng.uniform(0.4, 1.0)),
                    "gravity": rng.randint(1, 10),
                })
    insp_path, viol_path = out_dir / "osha_inspection.csv", out_dir / "osha_violation.csv"
    for path, rows in ((insp_path, inspections), (viol_path, violations)):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([k.upper() for k in rows[0]])
            w.writerows([list(r.values()) for r in rows])
    return insp_path, viol_path
