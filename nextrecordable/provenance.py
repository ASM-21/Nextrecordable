"""Where the loaded data came from, carried into every output.

`data/raw/SOURCES.json` declares, per raw file, what it is and where it was
obtained. `record()` hashes the files, stores the declaration in the
`data_provenance` table next to the data, and every report, API response and
PDF reads `notice()` from there. A database with no declaration is treated
as undeclared and says so: a run on synthetic or unverified files can never
pass for a run on OSHA's own.

SOURCES.json:
    {
      "kind": "official" | "third_party_copy" | "synthetic",
      "note": "free text shown in reports",
      "files": {
        "ita_summary/ita_summary_2023.csv": {
          "origin": "https://...", "snapshot": "through 2024-03-16",
          "kind": "third_party_copy", "note": "..."
        }
      }
    }
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb

KINDS = ("official", "third_party_copy", "synthetic", "undeclared")

NOTICES = {
    "official": None,
    "third_party_copy": (
        "DATA NOTICE: some input files are third-party copies of OSHA's published files, "
        "not downloads from osha.gov. See the data provenance table."
    ),
    "synthetic": (
        "SYNTHETIC DATA: these numbers come from generated test files, not from OSHA records. "
        "They describe no real establishment."
    ),
    "undeclared": (
        "UNDECLARED DATA: the source of the loaded files was not declared (no data/raw/SOURCES.json). "
        "Treat every number as synthetic until it is."
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _worst(kinds: list[str]) -> str:
    order = {"official": 0, "third_party_copy": 1, "undeclared": 2, "synthetic": 3}
    return max(kinds, key=lambda k: order.get(k, 2)) if kinds else "undeclared"


def declare(con: duckdb.DuckDBPyConnection, kind: str, note: str = "", files: list[dict] | None = None) -> None:
    """Write the provenance tables directly (tests and synthetic runs use this)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    con.execute("CREATE OR REPLACE TABLE data_provenance_summary (kind VARCHAR, note VARCHAR)")
    con.execute("INSERT INTO data_provenance_summary VALUES (?, ?)", [kind, note])
    con.execute(
        "CREATE OR REPLACE TABLE data_provenance (file VARCHAR, kind VARCHAR, origin VARCHAR, "
        "snapshot VARCHAR, note VARCHAR, bytes BIGINT, sha256 VARCHAR)"
    )
    for f in files or []:
        con.execute(
            "INSERT INTO data_provenance VALUES (?, ?, ?, ?, ?, ?, ?)",
            [f.get("file"), f.get("kind", kind), f.get("origin"), f.get("snapshot"), f.get("note"),
             f.get("bytes"), f.get("sha256")],
        )


def record(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> dict[str, Any]:
    """Read raw_dir/SOURCES.json, hash the declared files, store the result. Returns summary()."""
    raw_dir = Path(raw_dir)
    manifest = raw_dir / "SOURCES.json"
    if not manifest.exists():
        declare(con, "undeclared", f"no {manifest.name} in the raw data directory")
        return summary(con)
    spec = json.loads(manifest.read_text())
    default_kind = spec.get("kind", "undeclared")
    files = []
    for rel, meta in sorted(spec.get("files", {}).items()):
        path = raw_dir / rel
        files.append({
            "file": rel,
            "kind": meta.get("kind", default_kind),
            "origin": meta.get("origin"),
            "snapshot": meta.get("snapshot"),
            "note": meta.get("note") if path.exists() else "DECLARED BUT NOT PRESENT",
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        })
    kind = _worst([default_kind, *[f["kind"] for f in files]])
    declare(con, kind, spec.get("note", ""), files)
    return summary(con)


def summary(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """{'kind', 'note', 'notice', 'files': [...]}; undeclared when nothing was recorded."""
    has = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'data_provenance_summary'"
    ).fetchone()[0]
    if not has:
        return {"kind": "undeclared", "note": "", "notice": NOTICES["undeclared"], "files": []}
    kind, note = con.execute("SELECT kind, note FROM data_provenance_summary").fetchone()
    cols = ("file", "kind", "origin", "snapshot", "note", "bytes", "sha256")
    files = [dict(zip(cols, r, strict=True)) for r in con.execute(
        "SELECT file, kind, origin, snapshot, note, bytes, sha256 FROM data_provenance ORDER BY file"
    ).fetchall()]
    return {"kind": kind, "note": note, "notice": NOTICES.get(kind, NOTICES["undeclared"]), "files": files}


def notice(con: duckdb.DuckDBPyConnection) -> str | None:
    """The one-line warning every output must carry, or None for official data."""
    return summary(con)["notice"]


def snapshot_for(con: duckdb.DuckDBPyConnection, year: int) -> str | None:
    """Declared snapshot label of the ITA summary file for `year`, if any."""
    for f in summary(con)["files"]:
        if f["file"] and f"ita_summary_{year}." in f["file"]:
            return f["snapshot"]
    return None
