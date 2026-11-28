"""refresh-official: download checks, the copy comparison and the new SOURCES.json, all offline."""

from __future__ import annotations

import csv
import datetime as dt
import json
import zipfile
from pathlib import Path

import pytest

from nextrecordable import refresh
from nextrecordable.config import ITA_SUMMARY_URLS
from tests.synthetic import build_synthetic_dataset


def _zip_csv(csv_path: Path, dest: Path, *, drop_last_row: bool = False) -> Path:
    rows = list(csv.reader(csv_path.open(newline="")))
    if drop_last_row:
        rows = rows[:-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("readme.csv", "note\nmetadata only\n")      # real zips sometimes carry a small extra CSV
        with z.open(f"ITA_{csv_path.stem}.csv", "w") as f:
            f.write("\n".join(",".join(f'"{c}"' for c in r) for r in rows).encode())
    return dest


def test_snapshot_label_reads_the_file_name():
    today = dt.date(2026, 11, 25)
    assert refresh.snapshot_label(ITA_SUMMARY_URLS[2023], today) == "through 2024-12-31"
    assert refresh.snapshot_label(ITA_SUMMARY_URLS[2024], today) == "through 2025-12-31"
    assert refresh.snapshot_label(ITA_SUMMARY_URLS[2016], today) == "as published; downloaded 2026-11-25"


def test_verify_zip_rejects_a_saved_web_page(tmp_path):
    page = tmp_path / "ita_summary_2020.zip"
    page.write_bytes(b"<!DOCTYPE html><html><body>Access Denied</body></html>")
    with pytest.raises(refresh.RefreshError, match="web page"):
        refresh.verify_zip(page)


def test_verify_zip_rejects_missing_columns(tmp_path):
    bad = tmp_path / "ita_summary_2020.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("x.csv", "establishment_name,state\nAcme,PA\n")
    with pytest.raises(refresh.RefreshError, match="lacks columns"):
        refresh.verify_zip(bad)


def test_refresh_swaps_copies_for_official_files(tmp_path):
    synth = build_synthetic_dataset(tmp_path / "synth", n_establishments=120)
    raw, processed, reports = tmp_path / "raw", tmp_path / "processed", tmp_path / "reports"
    summary = raw / "ita_summary"
    summary.mkdir(parents=True)
    for year in (2019, 2023):                               # the copies being replaced
        (summary / f"ita_summary_{year}.csv").write_bytes(synth[year].read_bytes())
    (raw / "SOURCES.json").write_text(json.dumps({
        "kind": "third_party_copy",
        "files": {
            "ita_summary/ita_summary_2019.csv": {"origin": "copy", "kind": "third_party_copy"},
            "dol/OSHA_inspection.zip": {"origin": refresh.DOL_URLS["inspection"], "snapshot": "2026-11-15",
                                        "kind": "official"},
        },
    }))
    year_of = {v: k for k, v in ITA_SUMMARY_URLS.items()}
    # The official 2023 file has one more filing than the early copy did.
    out = refresh.run(
        [2019, 2023, 2024], train=False, raw_dir=raw, processed_dir=processed, reports_dir=reports,
        db_path=tmp_path / "db.duckdb", log=lambda _m: None, head=lambda _u: 200,
        fetch=lambda url, dest: _zip_csv(synth[year_of[url]], dest, drop_last_row=False),
    )
    # copies are out of the loader's sight, not deleted
    assert not list(summary.glob("ita_summary_*.csv"))
    assert sorted(p.name for p in (summary / refresh.COPIES_DIRNAME).iterdir()) == [
        "ita_summary_2019.csv", "ita_summary_2023.csv"]
    # every year was extracted where `load` will find it
    for y in (2019, 2023, 2024):
        assert list((processed / "ita_summary" / str(y)).glob("*.csv"))
    # identical files compare as identical
    cmp = json.loads((reports / "official_vs_copy.json").read_text())
    assert set(cmp["years"]) == {"2019", "2023"}
    assert cmp["years"]["2019"]["identical_counts"]
    # SOURCES.json: official, ITA zips declared, DOL entry kept, old file backed up
    src = json.loads((raw / "SOURCES.json").read_text())
    assert src["kind"] == "official"
    assert all(f["kind"] == "official" for f in src["files"].values())
    assert src["files"]["ita_summary/ita_summary_2023.zip"]["snapshot"] == "through 2024-12-31"
    assert "dol/OSHA_inspection.zip" in src["files"]
    assert "ita_summary/ita_summary_2019.csv" not in src["files"]
    assert json.loads((raw / "SOURCES.previous.json").read_text())["kind"] == "third_party_copy"
    assert out["moved_copies"]


def test_refresh_reports_a_differing_copy(tmp_path):
    synth = build_synthetic_dataset(tmp_path / "synth", n_establishments=80)
    raw = tmp_path / "raw"
    (raw / "ita_summary").mkdir(parents=True)
    (raw / "ita_summary" / "ita_summary_2023.csv").write_bytes(synth[2023].read_bytes())
    refresh.run(
        [2023], train=False, raw_dir=raw, processed_dir=tmp_path / "p", reports_dir=tmp_path / "r",
        db_path=tmp_path / "db.duckdb", log=lambda _m: None, head=lambda _u: 200,
        fetch=lambda url, dest: _zip_csv(synth[2023], dest, drop_last_row=True),
    )
    c = json.loads((tmp_path / "r" / "official_vs_copy.json").read_text())["years"]["2023"]
    assert not c["identical_counts"]
    assert c["official_minus_copy"]["rows"] == -1


def test_refresh_stops_before_touching_anything_when_a_url_is_gone(tmp_path):
    raw = tmp_path / "raw"
    (raw / "ita_summary").mkdir(parents=True)
    copy = raw / "ita_summary" / "ita_summary_2020.csv"
    copy.write_text("x\n")
    with pytest.raises(refresh.RefreshError, match="Unreachable"):
        refresh.run([2020], train=False, raw_dir=raw, processed_dir=tmp_path / "p", reports_dir=tmp_path / "r",
                    db_path=tmp_path / "db.duckdb", log=lambda _m: None, head=lambda _u: 403,
                    fetch=lambda *_a: pytest.fail("downloaded after a failed URL check"))
    assert copy.exists()
    assert not (raw / "SOURCES.json").exists()


def test_partial_refresh_keeps_other_copies_loaded_and_declared(tmp_path):
    """--year 2023 alone must not move the other copies or call the run official."""
    synth = build_synthetic_dataset(tmp_path / "synth", n_establishments=80)
    raw = tmp_path / "raw"
    summary = raw / "ita_summary"
    summary.mkdir(parents=True)
    for year in (2019, 2020, 2023):
        (summary / f"ita_summary_{year}.csv").write_bytes(synth[year].read_bytes())
    (raw / "SOURCES.json").write_text(json.dumps({
        "kind": "third_party_copy",
        "files": {"ita_summary/ita_summary_2019.csv": {"origin": "copy"},
                  "ita_summary/ita_summary_2023.csv": {"origin": "copy"}},
    }))
    refresh.run(
        [2023], train=False, raw_dir=raw, processed_dir=tmp_path / "p", reports_dir=tmp_path / "r",
        db_path=tmp_path / "db.duckdb", log=lambda _m: None, head=lambda _u: 200,
        fetch=lambda url, dest: _zip_csv(synth[2023], dest),
    )
    assert sorted(p.name for p in summary.glob("ita_summary_*.csv")) == ["ita_summary_2019.csv",
                                                                        "ita_summary_2020.csv"]
    src = json.loads((raw / "SOURCES.json").read_text())
    assert src["kind"] == "third_party_copy"
    assert src["files"]["ita_summary/ita_summary_2019.csv"]["kind"] == "third_party_copy"
    assert src["files"]["ita_summary/ita_summary_2020.csv"]["kind"] == "third_party_copy"  # was undeclared
    assert src["files"]["ita_summary/ita_summary_2023.zip"]["kind"] == "official"
    assert "ita_summary/ita_summary_2023.csv" not in src["files"]


def test_forced_download_replaces_an_existing_file(tmp_path, monkeypatch):
    """A refresh re-downloads over the old archive; rename() onto it fails on Windows."""
    from nextrecordable import download

    dest = tmp_path / "OSHA_inspection.zip"
    dest.write_bytes(b"old")

    class FakeResponse:
        def __init__(self):
            self.headers = {"content-length": "3"}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size):
            yield b"new"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(download.httpx, "stream", lambda *a, **k: FakeResponse())
    calls = []
    real_replace = Path.replace
    monkeypatch.setattr(Path, "replace", lambda self, t: calls.append(t) or real_replace(self, t))
    download.download_file("https://example.invalid/x.zip", dest, force=True)
    assert dest.read_bytes() == b"new"
    assert calls == [dest]
