"""Replace the third-party ITA copies with OSHA's own files and rerun the model.

    nextrecordable refresh-official                # 2016-2024 summary files, then load -> train -> falsify
    nextrecordable refresh-official --refresh-dol  # also re-download the DOL enforcement files
    nextrecordable refresh-official --no-train     # download, check and declare only

Run it on a machine that can reach osha.gov (the cloud build machines cannot;
see DATA-GAPS.md). Every step is safe to rerun: finished downloads are kept,
and nothing in data/ is deleted. What it does, in order:

1. HEAD every configured URL and stop if any is gone, before downloading anything.
2. Download each year's zip to data/raw/ita_summary/ita_summary_<year>.zip.
3. Check each zip: it opens, it holds a CSV, and the CSV has the columns the
   pipeline needs. A saved error page (OSHA's CDN sometimes answers 200 with
   HTML) fails here, not halfway through the build.
4. Compare each official year with the third-party copy it replaces (rows,
   distinct establishments, case and death totals) and write
   reports/official_vs_copy.json. Counts only; no establishment is named.
5. Move the copies to data/raw/ita_summary/third_party_copies/, where the
   loader no longer sees them.
6. Optionally refresh the DOL enforcement files from data.dol.gov.
7. Rewrite data/raw/SOURCES.json as "official" (the old one is kept as
   SOURCES.previous.json). Once no file is a third-party copy, the data
   notice drops out of every output by itself.
8. Run load, build, sanity, train and falsify.

After it finishes, `pytest tests/test_docs.py` names every number in
FINDINGS.md and the README that no longer matches reports/*.json.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import httpx

from . import config
from .download import HEADERS, download_file
from .load import extract_zip, find_csv

# data.dol.gov bulk files (DATA-GAPS.md, "Enforcement files").
DOL_URLS: dict[str, str] = {
    "inspection": "https://data.dol.gov/data-catalog/OSHA/inspection/OSHA_inspection.zip",
    "violation": "https://data.dol.gov/data-catalog/OSHA/violation/OSHA_violation.zip",
}

# One column from each group must be in every summary file, or the build cannot use it.
REQUIRED_COLUMN_GROUPS: dict[str, tuple[str, ...]] = {
    "name": config.NAME_COLUMN_CANDIDATES,
    "state": config.STATE_COLUMN_CANDIDATES,
    "zip": config.ZIP_COLUMN_CANDIDATES,
    "naics": config.NAICS_COLUMN_CANDIDATES,
    "employees": config.ANNUAL_EMPLOYEES_CANDIDATES,
    "hours": config.HOURS_CANDIDATES,
    "deaths": ("total_deaths",),
    "dafw": ("total_dafw_cases", "total_dafw_count"),
    "djtr": ("total_djtr_cases", "total_djtr_count"),
    "other": ("total_other_cases", "total_other_count"),
}

# Totals compared between an official file and the copy it replaces.
COMPARE_SUMS: dict[str, tuple[str, ...]] = {
    "total_deaths": ("total_deaths",),
    "total_dafw_cases": ("total_dafw_cases", "total_dafw_count"),
    "total_djtr_cases": ("total_djtr_cases", "total_djtr_count"),
    "total_other_cases": ("total_other_cases", "total_other_count"),
    "total_hours_worked": config.HOURS_CANDIDATES,
}

COPIES_DIRNAME = "third_party_copies"


class RefreshError(RuntimeError):
    """A check failed; nothing after it was changed."""


# ----------------------------------------------------------------- helpers


def snapshot_label(url: str, downloaded: dt.date) -> str:
    """'through 2024-12-31' when the file name says so, else the download date."""
    m = re.search(r"through[_ -](\d{1,2})-(\d{1,2})-(\d{4})", url)
    if m:
        month, day, year = (int(g) for g in m.groups())
        return f"through {year:04d}-{month:02d}-{day:02d}"
    return f"as published; downloaded {downloaded.isoformat()}"


def check_urls(urls: dict[str, str], *, head: Callable[[str], int] | None = None) -> dict[str, Any]:
    """{label: status} for every URL. head(url) -> status code; httpx by default."""
    def _head(url: str) -> int:
        return httpx.head(url, follow_redirects=True, timeout=30.0, headers=HEADERS).status_code

    head = head or _head
    out: dict[str, Any] = {}
    for label, url in urls.items():
        try:
            out[label] = head(url)
        except httpx.HTTPError as exc:
            out[label] = f"error: {exc.__class__.__name__}"
    return out


def _header(csv_bytes_head: bytes) -> list[str]:
    first = csv_bytes_head.decode("utf-8-sig", errors="replace").splitlines()[0]
    return [c.strip().strip('"').lower() for c in first.split(",")]


def verify_zip(path: Path) -> dict[str, Any]:
    """Raise RefreshError unless `path` is a readable zip holding a usable summary CSV."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        start = path.read_bytes()[:200].lower() if path.exists() else b""
        hint = " It looks like a web page, not a zip: delete it and retry." if b"<html" in start else ""
        raise RefreshError(f"{path.name} is not a zip file.{hint}") from exc
    with z:
        bad = z.testzip()
        if bad is not None:
            raise RefreshError(f"{path.name}: member {bad!r} is corrupt; delete the file and retry")
        csvs = [i for i in z.infolist() if i.filename.lower().endswith(".csv") and not i.is_dir()]
        if not csvs:
            raise RefreshError(f"{path.name} holds no CSV")
        biggest = max(csvs, key=lambda i: i.file_size)
        with z.open(biggest) as f:
            cols = set(_header(f.read(64 * 1024)))
    missing = [g for g, cands in REQUIRED_COLUMN_GROUPS.items() if not cols & set(cands)]
    if missing:
        raise RefreshError(f"{path.name} ({biggest.filename}) lacks columns for: {', '.join(missing)}")
    return {"csv": biggest.filename, "csv_bytes": biggest.file_size, "columns": len(cols)}


def profile_csv(csv_path: Path) -> dict[str, Any]:
    """Row count, distinct establishments and case totals of one summary CSV (counts only)."""
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE t AS SELECT * FROM read_csv(?, header = true, all_varchar = true, "
            "null_padding = true, ignore_errors = true, parallel = false)",
            [str(csv_path)],
        )
        cols = {r[1].lower(): r[1] for r in con.execute("PRAGMA table_info('t')").fetchall()}

        def pick(cands: tuple[str, ...]) -> str | None:
            return next((cols[c] for c in cands if c in cols), None)

        out: dict[str, Any] = {"rows": con.execute("SELECT COUNT(*) FROM t").fetchone()[0]}
        key = [pick(c) for c in (config.NAME_COLUMN_CANDIDATES, config.STATE_COLUMN_CANDIDATES,
                                 config.ZIP_COLUMN_CANDIDATES)]
        if all(key):
            expr = " || '|' || ".join(f"coalesce(upper(trim(\"{k}\")), '')" for k in key)
            out["distinct_name_state_zip"] = con.execute(f"SELECT COUNT(DISTINCT {expr}) FROM t").fetchone()[0]
        for name, cands in COMPARE_SUMS.items():
            col = pick(cands)
            if col:
                v = con.execute(f'SELECT SUM(TRY_CAST("{col}" AS DOUBLE)) FROM t').fetchone()[0]
                out[name] = None if v is None else round(float(v), 2)
        return out
    finally:
        con.close()


def compare(official: dict[str, Any], copy: dict[str, Any]) -> dict[str, Any]:
    """Field-by-field difference, official minus copy, and whether they agree exactly."""
    diff = {}
    for k, v in official.items():
        c = copy.get(k)
        if isinstance(v, int | float) and isinstance(c, int | float):
            diff[k] = round(v - c, 2)
    return {"official": official, "copy": copy, "official_minus_copy": diff,
            "identical_counts": all(d == 0 for d in diff.values())}


def stash_copies(summary_dir: Path) -> list[Path]:
    """Move ita_summary_<year>.csv copies out of the loader's sight. Returns new paths."""
    dest = summary_dir / COPIES_DIRNAME
    moved = []
    for p in sorted(summary_dir.glob("ita_summary_*.csv")):
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / p.name
        shutil.move(str(p), str(target))
        moved.append(target)
    return moved


def official_sources(raw_dir: Path, years: dict[int, str], downloaded: dt.date,
                     dol: dict[str, str] | None = None) -> dict[str, Any]:
    """The new SOURCES.json. Keeps declared DOL entries unless `dol` replaces them."""
    old_path = raw_dir / "SOURCES.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    files: dict[str, Any] = {}
    for year, url in sorted(years.items()):
        files[f"ita_summary/ita_summary_{year}.zip"] = {
            "origin": url, "snapshot": snapshot_label(url, downloaded), "kind": "official",
            "note": f"downloaded from osha.gov on {downloaded.isoformat()}",
        }
    if dol is not None:
        for stem, url in dol.items():
            files[f"dol/OSHA_{stem}.zip"] = {"origin": url, "snapshot": downloaded.isoformat(), "kind": "official"}
    else:
        for rel, meta in old.get("files", {}).items():
            if rel.startswith("dol/") and meta.get("kind", old.get("kind")) != "third_party_copy":
                files[rel] = {**meta, "kind": "official"}
    note = f"ITA Form 300A summary files downloaded from osha.gov ({downloaded.isoformat()})"
    if any(rel.startswith("dol/") for rel in files):
        note += "; DOL enforcement files from data.dol.gov"
    return {"kind": "official", "note": note + ".", "files": files}


def _dol_has_table(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return bool(con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'osha_inspection'"
        ).fetchone()[0])
    finally:
        con.close()


def _refresh_dol(raw_dir: Path, log: Callable[[str], None]) -> None:
    dol = raw_dir / "dol"
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for stem, url in DOL_URLS.items():
        zpath = download_file(url, dol / f"OSHA_{stem}.zip", force=True)
        with tempfile.TemporaryDirectory(dir=dol) as tmp:
            extract_zip(zpath, Path(tmp))
            chunks = sorted(Path(tmp).rglob("*.csv"))
            if not chunks:
                raise RefreshError(f"{zpath.name} holds no CSV")
            target = dol / stem
            if target.exists():
                (dol / "previous").mkdir(exist_ok=True)
                shutil.move(str(target), str(dol / "previous" / f"{stem}-{stamp}"))
            target.mkdir(parents=True)
            for c in chunks:
                shutil.move(str(c), str(target / c.name))
        log(f"  DOL {stem}: {len(chunks)} CSV chunks -> {target}")


# ------------------------------------------------------------------ driver


def run(
    years: list[int] | None = None,
    *,
    refresh_dol: bool = False,
    train: bool = True,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    reports_dir: Path | None = None,
    db_path: Path | None = None,
    log: Callable[[str], None] = print,
    head: Callable[[str], int] | None = None,
    fetch: Callable[[str, Path], Path] | None = None,
    run_cli: Callable[[list[str]], None] | None = None,
) -> dict[str, Any]:
    """Everything in the module docstring. Returns what it found; raises RefreshError on a failed check."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    db_path = Path(db_path or config.DB_PATH)
    years = sorted(years or config.ITA_SUMMARY_URLS)
    unknown = [y for y in years if y not in config.ITA_SUMMARY_URLS]
    if unknown:
        raise RefreshError(f"No URL configured for {unknown}; add it to ITA_SUMMARY_URLS in config.py")
    urls = {y: config.ITA_SUMMARY_URLS[y] for y in years}
    fetch = fetch or (lambda url, dest: download_file(url, dest))
    today = dt.date.today()
    t0 = time.monotonic()

    def _cli(args: list[str]) -> None:
        cmd = [sys.executable, "-m", "nextrecordable.cli", "--db", str(db_path), *args]
        log(f"$ nextrecordable {' '.join(args)}")
        subprocess.run(cmd, check=True)

    run_cli = run_cli or _cli

    # 1. URLs
    labels = {f"ITA {y}": u for y, u in urls.items()}
    if refresh_dol:
        labels |= {f"DOL {k}": u for k, u in DOL_URLS.items()}
    status = check_urls(labels, head=head)
    for label, s in status.items():
        log(f"  {label:16s} {s}  {labels[label]}")
    dead = [k for k, s in status.items() if not (isinstance(s, int) and s < 400)]
    if dead:
        raise RefreshError(
            f"Unreachable: {', '.join(dead)}. A 403 on every URL means osha.gov is refusing this network; "
            "otherwise fix the URL from https://www.osha.gov/itadata in nextrecordable/config.py."
        )

    # 2-3. download and check
    summary_dir = raw_dir / "ita_summary"
    checked: dict[int, dict[str, Any]] = {}
    for y, url in urls.items():
        dest = summary_dir / f"ita_summary_{y}.zip"
        log(f"Downloading {y} ...")
        fetch(url, dest)
        checked[y] = verify_zip(dest) | {"bytes": dest.stat().st_size}
        year_dir = processed_dir / "ita_summary" / str(y)
        if year_dir.exists():
            shutil.rmtree(year_dir)          # `load` reuses this extract, so it must be this zip's
        extract_zip(dest, year_dir)
        log(f"  ok: {checked[y]['csv']} ({checked[y]['csv_bytes']:,} bytes uncompressed)")

    # 4. compare with the copies they replace
    comparison: dict[str, Any] = {"generated": today.isoformat(), "years": {}}
    for y in years:
        copy_csv = summary_dir / f"ita_summary_{y}.csv"
        if not copy_csv.exists():
            continue
        off_csv = find_csv(processed_dir / "ita_summary" / str(y))
        comparison["years"][str(y)] = compare(profile_csv(off_csv), profile_csv(copy_csv))
        comparison["years"][str(y)]["official_snapshot"] = snapshot_label(urls[y], today)
        c = comparison["years"][str(y)]
        log(f"  {y}: official {c['official']['rows']:,} rows vs copy {c['copy']['rows']:,}"
            f"{'' if c['identical_counts'] else '  (differs)'}")
    if comparison["years"]:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "official_vs_copy.json").write_text(json.dumps(comparison, indent=2) + "\n")

    # 5. move the copies aside
    moved = stash_copies(summary_dir)
    for p in moved:
        log(f"  moved copy aside: {p}")

    # 6. DOL
    if refresh_dol:
        _refresh_dol(raw_dir, log)

    # 7. declare
    manifest = raw_dir / "SOURCES.json"
    if manifest.exists():
        shutil.copy2(manifest, raw_dir / "SOURCES.previous.json")
    sources = official_sources(raw_dir, urls, today, DOL_URLS if refresh_dol else None)
    manifest.write_text(json.dumps(sources, indent=2) + "\n")
    log(f"  wrote {manifest} ({len(sources['files'])} files, kind official)")

    # 8. rebuild
    if train:
        run_cli(["load"])
        dol_dir = raw_dir / "dol"
        if refresh_dol or not _dol_has_table(db_path):
            from .enforcement import find_dol_files

            insp, viol = find_dol_files(dol_dir)
            if insp and viol:
                run_cli(["load-enforcement", "--inspection", str(dol_dir / "inspection"),
                         "--violation", str(dol_dir / "violation")])
            else:
                log("  WARNING: no DOL files; rerun with --refresh-dol for enforcement features")
        run_cli(["build"])
        run_cli(["sanity"])
        run_cli(["train"])
        run_cli(["falsify"])

    minutes = (time.monotonic() - t0) / 60
    log(f"Done in {minutes:.0f} min. Next: `pytest tests/test_docs.py` lists the numbers to update "
        "in FINDINGS.md and the README; commit reports/ and those edits, never data/.")
    return {"status": status, "checked": checked, "comparison": comparison,
            "moved_copies": [str(p) for p in moved], "sources": sources}
