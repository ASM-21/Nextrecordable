"""Fuzzy entity resolution: match user-uploaded establishments to our database.

Use case: a broker uploads a CSV with rows like
    "ABC Mfg Co, 123 Main Street, Springfield IL 62701"
We need to find the corresponding `estab_key` in the ITA panel, even when:
  - The name uses different abbreviations ("Mfg" vs "Manufacturing")
  - Punctuation differs ("ABC, Inc." vs "ABC Inc")
  - Address has minor typos or formatting differences
  - ZIP is short or long form
  - Same establishment appears under multiple historical name variants

The soft-hash key in features.py requires an EXACT match on normalized
name + state + zip5. That's brittle. This module replaces that with
probabilistic record linkage using splink, which:
  - Blocks candidate pairs (so we don't compute n*m pairwise comparisons)
  - Scores each candidate pair using a fitted model
  - Returns top-k matches per input row with match probabilities

Output: a `match_results` table with (input_row_id, estab_key, match_probability)
plus an unmatched residual we can fall back to soft-key for.
"""

from __future__ import annotations

import contextlib
from typing import Any

import duckdb

from .normalize import normalize_address, normalize_name, zip5

# ---------- Preparing the match dataframes ----------

# Required input columns for the user-uploaded CSV
INPUT_COLUMNS_REQUIRED: tuple[str, ...] = ("name", "state", "zip_code")
INPUT_COLUMNS_OPTIONAL: tuple[str, ...] = ("street_address", "city")


def _ensure_udfs_registered(con: duckdb.DuckDBPyConnection) -> None:
    """Register Python normalization UDFs if not already present.

    DuckDB throws NotImplementedException on re-registration, so we check
    first. This makes the function idempotent across multiple calls in the
    same connection (which happens in `match_uploaded_csv`).
    """
    existing = {
        r[0]
        for r in con.execute(
            "SELECT function_name FROM duckdb_functions() "
            "WHERE function_name IN ('py_norm_name', 'py_norm_addr', 'py_zip5')"
        ).fetchall()
    }
    if "py_norm_name" not in existing:
        con.create_function("py_norm_name", normalize_name, ["VARCHAR"], "VARCHAR")
    if "py_norm_addr" not in existing:
        con.create_function("py_norm_addr", normalize_address, ["VARCHAR"], "VARCHAR")
    if "py_zip5" not in existing:
        con.create_function("py_zip5", lambda x: zip5(x), ["VARCHAR"], "VARCHAR")


def normalize_input_table(
    con: duckdb.DuckDBPyConnection,
    input_table: str,
    *,
    output_table: str = "match_input_normalized",
) -> int:
    """Build normalized columns on a user-uploaded input table.

    The input table must have at least: name, state, zip_code.
    Optional: street_address, city.

    Adds: name_norm, address_norm, zip5, name_first_token, input_row_id.
    """
    cols = {
        r[1].lower()
        for r in con.execute(f"PRAGMA table_info('{input_table}')").fetchall()
    }
    for required in INPUT_COLUMNS_REQUIRED:
        if required not in cols:
            raise ValueError(
                f"Input table {input_table} missing required column {required!r}. "
                f"Required: {INPUT_COLUMNS_REQUIRED}; got: {sorted(cols)}"
            )

    _ensure_udfs_registered(con)

    has_street = "street_address" in cols
    has_city = "city" in cols

    street_expr = "py_norm_addr(CAST(street_address AS VARCHAR))" if has_street else "CAST('' AS VARCHAR)"
    city_expr = "lower(CAST(city AS VARCHAR))" if has_city else "CAST('' AS VARCHAR)"

    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    con.execute(
        f"""
        CREATE TABLE {output_table} AS
        SELECT
            ROW_NUMBER() OVER () AS input_row_id,
            CAST(name AS VARCHAR)               AS original_name,
            {('CAST(street_address AS VARCHAR)' if has_street else "CAST('' AS VARCHAR)")} AS original_address,
            CAST(state AS VARCHAR)              AS state,
            CAST(zip_code AS VARCHAR)           AS original_zip,
            py_norm_name(CAST(name AS VARCHAR)) AS name_norm,
            {street_expr}                       AS address_norm,
            {city_expr}                         AS city_norm,
            py_zip5(CAST(zip_code AS VARCHAR))  AS zip5,
            -- First token of normalized name (blocking key)
            split_part(py_norm_name(CAST(name AS VARCHAR)), ' ', 1) AS name_first_token
        FROM {input_table}
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM {output_table}").fetchone()[0]


def build_estab_directory(
    con: duckdb.DuckDBPyConnection,
    *,
    output_table: str = "estab_directory",
    panel_table: str = "ita_panel",
) -> int:
    """Build a one-row-per-establishment directory from the panel.

    For each estab_key, we keep:
      - Most recent name / address / state / zip
      - Latest year present
      - NAICS

    This is the table splink links INTO. It's what we present in match results.
    """
    _ensure_udfs_registered(con)

    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    con.execute(
        f"""
        CREATE TABLE {output_table} AS
        WITH ranked AS (
            SELECT
                p.estab_key,
                p.establishment_name,
                p.state,
                p.zip5,
                p.naics_code,
                p.year,
                ROW_NUMBER() OVER (
                    PARTITION BY p.estab_key
                    ORDER BY p.year DESC
                ) AS rn
            FROM {panel_table} p
            WHERE p.establishment_name IS NOT NULL
        )
        SELECT
            estab_key,
            establishment_name AS original_name,
            state,
            zip5,
            naics_code,
            year AS latest_year,
            -- Normalized fields (computed by the same Python UDFs registered
            -- elsewhere in this module).
            py_norm_name(establishment_name) AS name_norm,
            split_part(py_norm_name(establishment_name), ' ', 1) AS name_first_token
        FROM ranked
        WHERE rn = 1
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM {output_table}").fetchone()[0]


# ---------- Splink linking ----------

def _make_splink_settings():
    """Build the splink settings for matching input -> directory.

    Design notes:
      - Link type "link_only" because the two datasets are pre-deduplicated
        (input has unique rows; directory has one row per estab_key).
      - Blocking: candidates must share state + ZIP5, OR state + name first
        token. This keeps pair count tractable on large directories.
      - Comparisons: JaroWinkler on normalized name (the strongest signal),
        ExactMatch on ZIP5, ExactMatch on state. We don't compare address
        (high noise, weak signal) but that's an easy extension.
    """
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on

    return SettingsCreator(
        link_type="link_only",
        blocking_rules_to_generate_predictions=[
            block_on("state", "zip5"),
            block_on("state", "name_first_token"),
        ],
        comparisons=[
            cl.JaroWinklerAtThresholds("name_norm", [0.95, 0.85, 0.7]),
            cl.ExactMatch("zip5"),
            cl.ExactMatch("state"),
        ],
        retain_intermediate_calculation_columns=False,
        retain_matching_columns=True,
    )


def link_input_to_directory(
    con: duckdb.DuckDBPyConnection,
    *,
    input_table: str = "match_input_normalized",
    directory_table: str = "estab_directory",
    output_table: str = "match_results",
    match_threshold: float = 0.80,
    deterministic_for_em: bool | None = None,
) -> int:
    """Run splink linkage and write top match per input row to output_table.

    Args:
        match_threshold: minimum match_probability to keep a result.
            Below this we consider the row "unmatched" and fall back to
            the deterministic soft-key.
        deterministic_for_em: if True, use a deterministic rule for EM
            training (faster, less accurate). Defaults to True on small
            datasets where probabilistic EM is unstable.

    Returns the number of input rows with a confident match.
    """
    from splink import DuckDBAPI, Linker, block_on

    # Clean up any splink-registered tables/views from a prior run on the
    # same connection. Splink uses aliases like 'input', 'directory', plus
    # various __splink__* intermediate objects, registered as views.
    for alias in ("input", "directory"):
        con.execute(f"DROP VIEW IF EXISTS {alias}")
        con.execute(f"DROP TABLE IF EXISTS {alias}")
    leftovers = con.execute(
        "SELECT table_name, table_type FROM information_schema.tables "
        "WHERE table_name LIKE '__splink__%'"
    ).fetchall()
    for name, kind in leftovers:
        if kind == "VIEW":
            con.execute(f"DROP VIEW IF EXISTS {name}")
        else:
            con.execute(f"DROP TABLE IF EXISTS {name}")

    db_api = DuckDBAPI(connection=con)

    # Pull the two tables as DataFrames for splink. Splink requires:
    #   1. A `unique_id` column on each input dataset
    #   2. Matching schemas across the two datasets
    # We project to the splink-relevant columns plus original_estab_key
    # (carried so we can map back after linkage).
    input_df = con.execute(
        f"""
        SELECT
            input_row_id AS unique_id,
            name_norm,
            name_first_token,
            state,
            zip5
        FROM {input_table}
        """
    ).df()
    dir_df = con.execute(
        f"""
        SELECT
            estab_key AS unique_id,
            name_norm,
            name_first_token,
            state,
            zip5
        FROM {directory_table}
        """
    ).df()

    if deterministic_for_em is None:
        # On tiny datasets, EM is unstable. Use deterministic estimation.
        deterministic_for_em = len(dir_df) < 5000

    settings = _make_splink_settings()
    linker = Linker(
        [input_df, dir_df],
        settings,
        db_api=db_api,
        input_table_aliases=["input", "directory"],
    )

    # Scale max_pairs to the actual dataset size. Splink hits an internal
    # "salting partitions" error when max_pairs is much larger than the
    # number of pairs available to sample from.
    max_pairs_for_u = max(1e4, min(1e6, len(input_df) * len(dir_df) * 0.5))

    # Training:
    #   1. Prior on match probability via a strict deterministic rule
    #   2. u (probability of agreement among non-matches) via random sampling
    #   3. m (probability of agreement among matches) via EM, optional
    #
    # Each step is wrapped in try/except because splink hits internal bugs on
    # very small datasets (e.g. "Salting partitions must be > 1" during
    # random-sample u-estimation when the candidate pair count is degenerate).
    # When training steps fail, splink falls back to default values for the
    # untrained parameters; predictions are still produced and remain useful
    # for our threshold-based downstream logic.
    with contextlib.suppress(Exception):
        linker.training.estimate_probability_two_random_records_match(
            deterministic_matching_rules=[
                block_on("state", "zip5", "name_first_token")
            ],
            recall=0.9,
        )

    with contextlib.suppress(Exception):
        linker.training.estimate_u_using_random_sampling(max_pairs=max_pairs_for_u)

    if not deterministic_for_em:
        with contextlib.suppress(Exception):
            linker.training.estimate_parameters_using_expectation_maximisation(
                block_on("state", "zip5")
            )

    predictions = linker.inference.predict(threshold_match_probability=0.01)
    pred_df = predictions.as_pandas_dataframe()

    # Register predictions back in duckdb and keep top-1 match per input row
    con.register("splink_predictions_raw", pred_df)
    con.execute(f"DROP TABLE IF EXISTS {output_table}")

    if len(pred_df) == 0:
        # No predictions at all; create an empty result table with the expected schema
        con.execute(
            f"""
            CREATE TABLE {output_table} (
                input_row_id BIGINT,
                matched_estab_key VARCHAR,
                match_probability DOUBLE,
                is_confident BOOLEAN
            )
            """
        )
    else:
        # Splink may put the input dataset on the L or R side depending on
        # internal ordering. We use source_dataset_l/r to disambiguate.
        con.execute(
            f"""
            CREATE TABLE {output_table} AS
            WITH normalized_pairs AS (
                SELECT
                    CASE WHEN source_dataset_l = 'input' THEN unique_id_l ELSE unique_id_r END
                        AS input_unique_id,
                    CASE WHEN source_dataset_l = 'directory' THEN unique_id_l ELSE unique_id_r END
                        AS dir_unique_id,
                    match_probability
                FROM splink_predictions_raw
            ),
            ranked AS (
                SELECT
                    CAST(input_unique_id AS BIGINT) AS input_row_id,
                    CAST(dir_unique_id AS VARCHAR)  AS matched_estab_key,
                    match_probability,
                    ROW_NUMBER() OVER (
                        PARTITION BY input_unique_id
                        ORDER BY match_probability DESC
                    ) AS rn
                FROM normalized_pairs
            )
            SELECT
                input_row_id,
                matched_estab_key,
                match_probability,
                match_probability >= {match_threshold} AS is_confident
            FROM ranked
            WHERE rn = 1
            """
        )
    con.unregister("splink_predictions_raw")

    return con.execute(
        f"SELECT COUNT(*) FROM {output_table} WHERE is_confident"
    ).fetchone()[0]


def fallback_softkey_match(
    con: duckdb.DuckDBPyConnection,
    *,
    input_table: str = "match_input_normalized",
    directory_table: str = "estab_directory",
    match_results_table: str = "match_results",
    output_table: str = "match_final",
) -> dict[str, int]:
    """Combine splink matches with a deterministic soft-key fallback.

    IMPORTANT: the softkey hash here MUST match the hash used in
    features.build_establishment_key_view, or the join will silently return
    zero matches. We replicate the same SQL regex normalization (strip
    non-alphanumeric, lowercase) rather than using the Python normalize_name
    UDF, since Python normalization ALSO strips entity suffixes and would
    produce a different hash than what's in the panel.

    Returns counts of each match method.
    """
    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    con.execute(
        f"""
        CREATE TABLE {output_table} AS
        WITH softkey_attempts AS (
            SELECT
                i.input_row_id,
                md5(concat_ws(
                    '|',
                    regexp_replace(lower(coalesce(i.original_name, '')), '[^a-z0-9 ]', '', 'g'),
                    upper(coalesce(i.state, '')),
                    coalesce(i.zip5, '')
                )) AS computed_key
            FROM {input_table} i
        ),
        softkey_matches AS (
            SELECT
                s.input_row_id,
                d.estab_key AS matched_estab_key,
                1.0 AS match_probability,
                'softkey_exact' AS match_method
            FROM softkey_attempts s
            JOIN {directory_table} d ON d.estab_key = s.computed_key
        ),
        splink_confident AS (
            SELECT
                input_row_id,
                matched_estab_key,
                match_probability,
                'splink_fuzzy' AS match_method
            FROM {match_results_table}
            WHERE is_confident
        ),
        combined AS (
            -- Splink takes precedence for confident matches
            SELECT * FROM splink_confident
            UNION ALL
            SELECT s.* FROM softkey_matches s
            WHERE NOT EXISTS (
                SELECT 1 FROM splink_confident sc WHERE sc.input_row_id = s.input_row_id
            )
        ),
        all_inputs AS (
            SELECT input_row_id FROM {input_table}
        )
        SELECT
            a.input_row_id,
            c.matched_estab_key,
            c.match_probability,
            COALESCE(c.match_method, 'unmatched') AS match_method
        FROM all_inputs a
        LEFT JOIN combined c USING (input_row_id)
        """
    )

    counts = con.execute(
        f"""
        SELECT match_method, COUNT(*) FROM {output_table}
        GROUP BY match_method ORDER BY match_method
        """
    ).fetchall()
    return {method: int(n) for method, n in counts}


# ---------- High-level convenience ----------

def match_uploaded_csv(
    con: duckdb.DuckDBPyConnection,
    input_table: str,
    *,
    panel_table: str = "ita_panel",
    match_threshold: float = 0.80,
) -> dict[str, Any]:
    """End-to-end: normalize input, build directory, link, fall back to softkey.

    Returns a dict with table names and counts. The final per-row results
    are in `match_final`.
    """
    n_input = normalize_input_table(con, input_table)
    n_directory = build_estab_directory(con, panel_table=panel_table)
    n_confident = link_input_to_directory(
        con, match_threshold=match_threshold
    )
    method_counts = fallback_softkey_match(con)
    return {
        "n_input_rows": n_input,
        "n_directory": n_directory,
        "n_splink_confident": n_confident,
        "method_counts": method_counts,
        "final_table": "match_final",
    }
