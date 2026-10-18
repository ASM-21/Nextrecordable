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
