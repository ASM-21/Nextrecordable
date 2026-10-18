"""The establishment key: SQL and Python must agree, or joins silently fail."""

import random

import duckdb

from nextrecordable.keys import build_keyed_table, python_key, state_sql


def test_sql_key_matches_python_key_on_messy_inputs():
    rng = random.Random(3)
    names = ["Acme Forge, Inc.", "ACME FORGE INC", "The Acme Forge Company", "O'Brien & Sons LLC",
             "Elm City Castings", "  Gateway  Steel  ", "S&S Machine Co.", None, ""]
    states = ["IL", " il", "ct", "NJ ", "Ohio", None]
    zips = ["61602", "06511", 6511, "6511-1234", "07030.0", "123", None, "02139-4307"]
    rows = [(rng.choice(names), rng.choice(states), rng.choice(zips)) for _ in range(400)]
    con = duckdb.connect()
    con.execute("CREATE TABLE src (n VARCHAR, s VARCHAR, z VARCHAR)")
    con.executemany("INSERT INTO src VALUES (?, ?, ?)",
                    [(n, s, None if z is None else str(z)) for n, s, z in rows])
    build_keyed_table(con, source="src", output="keyed", name_col="n", state_col="s", zip_col="z")
    got = con.execute("SELECT n, s, z, estab_key FROM keyed").fetchall()
    for n, s, z, key in got:
        assert key == python_key(n, s, z), (n, s, z)


def test_state_sql_matches_python():
    from nextrecordable.normalize import state_code

    con = duckdb.connect()
    for v in ["IL", " il ", "ct", "Ohio", "", None]:
        assert con.execute(f"SELECT {state_sql('?')}", [v]).fetchone()[0] == state_code(v)


def test_leading_zero_zip_and_spelling_variants_share_a_key():
    assert python_key("Elm City Castings Inc", "CT", "06511") == python_key("ELM CITY CASTINGS", "ct", 6511)
    assert python_key("Acme Mfg Co", "IL", "61602-1234") == python_key("Acme Manufacturing", "IL", "61602")
    assert python_key("Acme", "IL", "61602") != python_key("Acme", "IL", "61603")
