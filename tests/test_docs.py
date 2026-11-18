"""Prose guards: the README can't quietly drop a caveat or cite things that don't exist."""

from __future__ import annotations

import re
from pathlib import Path

from nextrecordable.cli import app
from nextrecordable.report import STANDARD_CAVEATS

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()


def test_required_statements_present():
    for phrase in (
        "Not affiliated with",           # independence from OSHA / DOL
        "undercount",                     # reported vs true injuries
        "under 20 employees",             # coverage
        "legal review",                   # republishing scores about named employers
        "none on record",                 # meaning of zero inspections
    ):
        assert phrase in README, phrase


def test_status_line_matches_reality():
    """Until a real-data METRICS.md is committed, the README must say so."""
    if not (ROOT / "reports" / "METRICS.md").exists():
        assert "has not yet been run on real OSHA data" in README


def test_report_caveats_cover_the_basics():
    text = " ".join(STANDARD_CAVEATS).lower()
    for phrase in ("undercount", "size-adjusted", "not that the site was inspected"):
        assert phrase in text, phrase


def test_cli_commands_in_readme_exist():
    names = {c.name or c.callback.__name__.replace("_", "-") for c in app.registered_commands}
    names |= {g.name for g in app.registered_groups}
    mentioned = set(re.findall(r"(?:^|`)nextrecordable ([a-z][a-z-]+)", README, flags=re.M))
    assert mentioned <= names, mentioned - names


def test_tests_cited_in_readme_exist():
    cited = set(re.findall(r"`(test_[a-z0-9_]+)`", README))
    source = "\n".join(p.read_text() for p in (ROOT / "tests").glob("test_*.py"))
    missing = {t for t in cited if f"def {t}(" not in source}
    assert not missing, missing


# ------------------------------------------------- numbers quoted from results

import json  # noqa: E402

REPORTS = ROOT / "reports"
NUMBER = re.compile(r"(?<![\w./-])-?\d[\d,]*(?:\.\d+)?%?")

# Numbers in the pinned documents that are not results: thresholds and facts
# reproduced by tests/test_real_data_quirks.py and the directive data file.
STATIC = {
    "1.7", "3.4", "16.8", "4.0", "0.02", "1,923", "100,000", "40,000", "1,000,000",
    "1,000", "2,000", "5,000", "25,000", "10%", "1-19", "20-49", "50-99", "100-249", "250-999", "1,000+",
    "36", "652",
}


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    nxt = text.find("\n## ", start + len(heading))
    nxt2 = text.find("\n### ", start + len(heading))
    ends = [e for e in (nxt, nxt2) if e != -1]
    return text[start: min(ends) if ends else len(text)]


def _allowed() -> set[str]:
    out: set[str] = set(STATIC)

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, bool) or v is None:
            return
        elif isinstance(v, int | float):
            f = float(v)
            out.update({f"{f:.3f}", f"{f:.2f}", f"{f:.1f}", f"{100 * f:.1f}%", f"{100 * f:.0f}%"})
            if f == int(f):
                out.update({f"{int(f):,}", str(int(f))})

    for name in ("metrics.json", "falsification.json"):
        walk(json.loads((REPORTS / name).read_text()))
    for t in json.loads((ROOT / "nextrecordable" / "data" / "tiers.json").read_text())["tiers"]:
        walk(t)
    return out


def _unpinned(text: str) -> list[str]:
    allowed = _allowed()
    found = []
    for tok in NUMBER.findall(text):
        if not any(c in tok for c in ".,%") or tok.endswith((",", ".")):
            continue                              # plain small integers and years: prose, not results
        if tok not in allowed and tok.lstrip("-") not in allowed:
            found.append(tok)
    return found


def test_findings_numbers_come_from_the_results():
    assert not _unpinned((ROOT / "FINDINGS.md").read_text())


def test_readme_results_and_pricing_come_from_the_results():
    text = _section(README, "## Results on real data") + _section(README, "### Keys, tiers and pricing")
    assert not _unpinned(text)


def test_readme_limits_match_code():
    from nextrecordable.api import ANONYMOUS_DAILY_LOOKUPS, MAX_BATCH_ROWS, MAX_UPLOAD_BYTES

    assert f"{MAX_UPLOAD_BYTES // 1_048_576} MB and {MAX_BATCH_ROWS:,} rows per upload" in README
    assert f"{ANONYMOUS_DAILY_LOOKUPS} a day per client" in README


def test_metrics_md_carries_the_data_notice_for_its_provenance():
    payload = json.loads((REPORTS / "metrics.json").read_text())
    notice = payload["data_provenance"]["notice"]
    md = (REPORTS / "METRICS.md").read_text()
    assert (notice in md) if notice else ("NOTICE" not in md.split("## Data")[0])


def test_pin_check_catches_a_wrong_number():
    assert _unpinned("AUC 0.861 and 49.9% captured") == ["0.861", "49.9%"]


def test_package_data_is_not_gitignored():
    """`data/` in .gitignore once also matched nextrecordable/data/, so the SST directive file and the
    tiers were neither committed nor shipped in the wheel (hatch honors .gitignore)."""
    import shutil
    import subprocess

    import pytest

    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    for rel in ("nextrecordable/data/sst_cpl_02_01_067.json", "nextrecordable/data/tiers.json"):
        ignored = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode == 0
        assert not ignored, rel
