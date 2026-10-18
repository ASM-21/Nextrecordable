"""Record linkage: find the establishment a name + address refers to.

A small Fellegi-Sunter matcher with FIXED weights, run entirely in DuckDB.
It replaces a per-request splink model, which had three problems: it trained
on whatever rows a request happened to contain (one typo'd lookup trained a
model on one row, and one-typo lookups matched 12 of 40 times), it needed
shared temp views that broke under concurrent requests, and splink 5.0
removed the API it was written against.

How a match is scored
---------------------
Candidates share the state and either the ZIP or the first name token
(blocking). Each candidate gets a weight in bits (log2 likelihood ratio):

  name (normalized)
      identical +9 | one edit apart (a typo) +7 | Jaro-Winkler >= 0.92  +3
      | >= 0.80  0 | else -8
  numbers in the name ("plant 2", "store 1234")
      both present, equal +2 | both present, different -12 | otherwise 0
  ZIP  same +4 | same first 3 digits 0 | different -3 | missing 0
  street address (both present)
      Jaro-Winkler >= 0.95  +3 | >= 0.85  +1 | else -2

match probability = 1 / (1 + 2^-(PRIOR_BITS + weight)). A row is matched
when PRIOR_BITS + weight >= MATCH_BITS (probability 0.8) and the best
candidate beats the runner-up by AMBIGUITY_MARGIN_BITS; a close runner-up
makes it "ambiguous" rather than a coin flip. Some reference points:
a one-letter typo with the right ZIP scores 3 bits (0.89); the exact name
in a neighboring ZIP scores 1 bit (0.67, unmatched unless the street agrees);
"Summit Castings" vs "Summit Coatings" in one ZIP scores -1 bit (0.33).

Why edits and not Jaro-Winkler for the strong band: in a long name, swapping
one word for a similar one ("Castings" -> "Coatings") still scores
Jaro-Winkler above 0.95, the same as a typo. The match-quality harness
caught exactly that: a lookup for a company that doesn't file came back as
its neighbor. A typo is one edit; a different word is at least two. The weights were set by hand and checked against the
perturbation harness in tests/test_match_quality.py, which pins per-case
accuracy floors and hard-negative false-match ceilings.

Every function takes the connection or cursor to run on. Per-request work
uses TEMP tables, which are private to a DuckDB connection, so concurrent
requests on separate cursors never see each other's rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import duckdb
import pandas as pd

from .normalize import (
    name_first_token,
    name_numbers,
    normalize_address,
    normalize_name,
    state_code,
    zip5,
)

PRIOR_BITS = -8.0
MATCH_BITS = 2.0  # log2(0.8 / 0.2): match probability 0.8
MATCH_THRESHOLD = 0.80
AMBIGUITY_MARGIN_BITS = 2.0

WEIGHT_SQL = """
    (CASE WHEN l_name = r_name THEN 9
          WHEN name_dl <= 1 AND length(l_name) >= 5 THEN 7
          WHEN name_jw >= 0.92 THEN 3 WHEN name_jw >= 0.80 THEN 0 ELSE -8 END)
  + (CASE WHEN l_nums <> '' AND r_nums <> '' AND l_nums = r_nums THEN 2
          WHEN l_nums <> '' AND r_nums <> '' THEN -12 ELSE 0 END)
  + (CASE WHEN l_zip = '' OR r_zip = '' THEN 0 WHEN l_zip = r_zip THEN 4
          WHEN substr(l_zip, 1, 3) = substr(r_zip, 1, 3) THEN 0 ELSE -3 END)
  + (CASE WHEN l_addr = '' OR r_addr = '' THEN 0 WHEN addr_jw >= 0.95 THEN 3
          WHEN addr_jw >= 0.85 THEN 1 ELSE -2 END)
"""

LEFT_COLUMNS = ("rec_id", "name_norm", "name_first", "name_nums", "state", "zip5", "address_norm")


def normalize_inputs(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalize user rows (name, state, zip_code, optional street_address/city).

    Returns the linker's left-hand frame; rec_id is 1-based input order.
    """
    out: dict[str, list] = {c: [] for c in (*LEFT_COLUMNS, "original_name")}
    for i, row in enumerate(rows, start=1):
        name = row.get("name")
        norm = normalize_name(name)
        out["rec_id"].append(i)
        out["original_name"].append("" if name is None else str(name))
        out["name_norm"].append(norm)
        out["name_first"].append(name_first_token(norm))
        out["name_nums"].append(name_numbers(norm))
        out["state"].append(state_code(row.get("state")))
        out["zip5"].append(zip5(row.get("zip_code")))
        out["address_norm"].append(normalize_address(row.get("street_address")))
    return pd.DataFrame({k: pd.Series(v, dtype="int64" if k == "rec_id" else "object") for k, v in out.items()})


def load_left(con, frame: pd.DataFrame, table: str = "_match_left") -> None:
    """Put a normalized frame into a TEMP table on `con`."""
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(
        f"""CREATE TEMP TABLE {table} (rec_id BIGINT, name_norm VARCHAR, name_first VARCHAR,
            name_nums VARCHAR, state VARCHAR, zip5 VARCHAR, address_norm VARCHAR,
            original_name VARCHAR)"""
    )
    if len(frame):
        con.register("_match_left_df", frame)
        con.execute(
            f"INSERT INTO {table} SELECT rec_id, name_norm, name_first, name_nums, state, "
            f"zip5, address_norm, original_name FROM _match_left_df"
        )
        con.unregister("_match_left_df")


def build_directory(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_table: str = "ita_panel",
    output: str = "estab_directory",
) -> int:
    """One row per estab_key (latest filing): what lookups are matched against."""
    con.execute(f"DROP TABLE IF EXISTS {output}")
    con.execute(
        f"""
        CREATE TABLE {output} AS
        SELECT estab_key, establishment_name, street_address, city, state, zip5,
               name_norm, name_first, name_nums, coalesce(address_norm, '') AS address_norm,
               naics_code, year AS latest_year
        FROM {panel_table}
        WHERE name_norm <> ''
        QUALIFY ROW_NUMBER() OVER (PARTITION BY estab_key ORDER BY year DESC) = 1
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM {output}").fetchone()[0]


def link(
    con,
    *,
    left: str = "_match_left",
    directory: str = "estab_directory",
    output: str = "_match_best",
    candidates_output: str | None = "_match_candidates",
    match_bits: float = MATCH_BITS,
    temp: bool = True,
) -> dict[str, int]:
    """Score candidate pairs and keep the best per left row.

    `output` gets one row per left rec_id: best estab_key (NULL if no
    candidate), weight, probability, runner-up weight and status in
    {'matched', 'ambiguous', 'unmatched'}. When `candidates_output` is set it
    keeps every scored pair (for "did you mean" suggestions).
    Returns counts by status.
    """
    kind = "TEMP TABLE" if temp else "TABLE"
    con.execute("DROP TABLE IF EXISTS _match_pairs")
    con.execute(
        f"""
        CREATE TEMP TABLE _match_pairs AS
        SELECT l.rec_id, d.estab_key
        FROM {left} l JOIN {directory} d
          ON l.state = d.state AND l.zip5 = d.zip5
        WHERE l.zip5 <> '' AND l.name_norm <> ''
        UNION
        SELECT l.rec_id, d.estab_key
        FROM {left} l JOIN {directory} d
          ON l.state = d.state AND l.name_first = d.name_first
        WHERE l.name_first <> ''
        """
    )
    scored_sql = f"""
        SELECT rec_id, estab_key, name_jw, name_dl, addr_jw, ({WEIGHT_SQL}) AS weight
        FROM (
            SELECT p.rec_id, p.estab_key,
                   jaro_winkler_similarity(l.name_norm, d.name_norm) AS name_jw,
                   damerau_levenshtein(l.name_norm, d.name_norm) AS name_dl,
                   l.name_norm AS l_name, d.name_norm AS r_name,
                   CASE WHEN l.address_norm <> '' AND d.address_norm <> ''
                        THEN jaro_winkler_similarity(l.address_norm, d.address_norm) END AS addr_jw,
                   l.name_nums AS l_nums, d.name_nums AS r_nums,
                   l.zip5 AS l_zip, d.zip5 AS r_zip,
                   l.address_norm AS l_addr, d.address_norm AS r_addr
            FROM _match_pairs p
            JOIN {left} l USING (rec_id)
            JOIN {directory} d USING (estab_key)
        )
    """
    if candidates_output:
        con.execute(f"DROP TABLE IF EXISTS {candidates_output}")
        con.execute(f"CREATE {kind} {candidates_output} AS {scored_sql}")
        source = candidates_output
    else:
        con.execute("DROP TABLE IF EXISTS _match_scored")
        con.execute(f"CREATE TEMP TABLE _match_scored AS {scored_sql}")
        source = "_match_scored"

    prob = f"1.0 / (1.0 + pow(2.0, -({PRIOR_BITS} + weight)))"
    con.execute(f"DROP TABLE IF EXISTS {output}")
    con.execute(
        f"""
        CREATE {kind} {output} AS
        WITH ranked AS (
            SELECT rec_id, estab_key, weight, name_jw,
                   ROW_NUMBER() OVER (PARTITION BY rec_id ORDER BY weight DESC, name_jw DESC, estab_key) AS rk,
                   LEAD(weight) OVER (PARTITION BY rec_id ORDER BY weight DESC, name_jw DESC, estab_key) AS runner_up
            FROM {source}
        ),
        best AS (SELECT * FROM ranked WHERE rk = 1)
        SELECT
            l.rec_id,
            b.estab_key,
            b.weight,
            b.runner_up,
            CASE WHEN b.weight IS NULL THEN 0.0 ELSE {prob} END AS match_probability,
            CASE
                WHEN b.weight IS NULL OR {PRIOR_BITS} + b.weight < {match_bits} THEN 'unmatched'
                WHEN b.runner_up IS NOT NULL AND b.weight - b.runner_up < {AMBIGUITY_MARGIN_BITS}
                    THEN 'ambiguous'
                ELSE 'matched'
            END AS status
        FROM {left} l LEFT JOIN best b USING (rec_id)
        """
    )
    for t in ("_match_pairs", "_match_scored"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    rows = con.execute(f"SELECT status, COUNT(*) FROM {output} GROUP BY 1").fetchall()
    return {s: int(n) for s, n in rows}


def suggestions(con, rec_id: int, *, candidates: str = "_match_candidates",
                directory: str = "estab_directory", k: int = 3,
                min_probability: float = 0.05) -> list[dict[str, Any]]:
    """Top-k scored candidates for one input row, for "did you mean" responses.

    Candidates below `min_probability` are dropped: a block can contain an
    unrelated business that merely shares the ZIP, and listing it as "closest"
    invites a wrong click.
    """
    prob = f"1.0 / (1.0 + pow(2.0, -({PRIOR_BITS} + c.weight)))"
    rows = con.execute(
        f"""
        SELECT d.establishment_name, d.street_address, d.city, d.state, d.zip5, {prob}
        FROM {candidates} c JOIN {directory} d USING (estab_key)
        WHERE c.rec_id = ? AND {prob} >= {float(min_probability)}
        ORDER BY c.weight DESC, c.name_jw DESC
        LIMIT {int(k)}
        """,
        [rec_id],
    ).fetchall()
    keys = ("establishment_name", "street_address", "city", "state", "zip5", "match_probability")
    return [dict(zip(keys, r, strict=True)) for r in rows]


def match_rows(con, rows: Iterable[Mapping[str, Any]], *, directory: str = "estab_directory") -> pd.DataFrame:
    """Normalize and link user rows. Returns one row per input with match fields.

    Runs on `con`'s TEMP namespace; pass a per-request cursor from a server.
    """
    frame = normalize_inputs(rows)
    load_left(con, frame)
    link(con, directory=directory)
    return con.execute(
        """
        SELECT l.rec_id, l.original_name, b.estab_key, b.match_probability, b.status
        FROM _match_left l JOIN _match_best b USING (rec_id)
        ORDER BY l.rec_id
        """
    ).df()
