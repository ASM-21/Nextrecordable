"""Normalization rules (the single source of truth for names, addresses, ZIPs)."""

import random

import pytest

from osha_risk.normalize import (
    name_first_token,
    name_numbers,
    normalize_address,
    normalize_name,
    state_code,
    zip5,
)


class TestNormalizeName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ABC Manufacturing Inc", "abc manufacturing"),
            ("ABC Manufacturing, LLC", "abc manufacturing"),
            ("ABC Mfg Co Inc", "abc manufacturing"),
            ("ABC MFG", "abc manufacturing"),
            ("Smith & Jones LLC", "smith and jones"),
            ("Smith&Jones", "smith and jones"),
            ("The Acme Company", "acme"),
            ("McDonald's Corp", "mcdonalds"),
            ("McDonald" + chr(0x2019) + "s Corp", "mcdonalds"),
            ("Acme Inc Holdings", "acme inc holdings"),
            ("S & S Machine", "s and s machine"),
            ("Gateway Castings Corporaiton", "gateway castings"),
            ("Acme Incorperated", "acme"),
            ("Acme Compton", "acme compton"),
            ("Inc", "inc"),
        ],
    )
    def test_examples(self, raw, expected):
        assert normalize_name(raw) == expected

    def test_missing(self):
        assert normalize_name(None) == ""
        assert normalize_name(float("nan")) == ""
        assert normalize_name("") == ""

    def test_idempotent_on_random_names(self):
        rng = random.Random(0)
        parts = ["Acme", "MFG", "Co.", "&", "Inc", "the", "O'Brien", "Plant #2", "Corporation", "LLC"]
        for _ in range(500):
            s = " ".join(rng.choice(parts) for _ in range(rng.randint(1, 6)))
            once = normalize_name(s)
            assert normalize_name(once) == once, s


class TestNormalizeAddress:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("123 Main Street", "123 main st"),
            ("100 North Main St", "100 n main st"),
            ("123 Main St., Suite #100", "123 main st ste 100"),
            ("789 Industrial Boulevard", "789 industrial blvd"),
        ],
    )
    def test_examples(self, raw, expected):
        assert normalize_address(raw) == expected

    def test_missing(self):
        assert normalize_address(None) == ""


class TestZip5:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("12345", "12345"),
            ("12345-6789", "12345"),
            ("123456789", "12345"),
            (12345, "12345"),
            ("ZIP: 12345-6789!", "12345"),
            # A CT ZIP after an integer round trip (DuckDB sniffing, DOL files, Excel)
            (6511, "06511"),
            ("6511", "06511"),
            ("6511.0", "06511"),
            ("6511-1234", "06511"),
            (65111234, "06511"),
            ("501", "00501"),
            ("12", ""),
            ("1234567890", ""),
            (None, ""),
            (float("nan"), ""),
        ],
    )
    def test_examples(self, raw, expected):
        assert zip5(raw) == expected

    def test_integer_round_trip_is_lossless(self):
        # Every 5-digit ZIP survives being stored as an integer.
        for z in ("00501", "01001", "06511", "07030", "10001", "99950"):
            assert zip5(int(z)) == z
            assert zip5(f"{int(z)}-1234") == z


def test_name_numbers():
    assert name_numbers("acme plant 02 store 117") == "117 2"
    assert name_numbers("acme forge") == ""


def test_first_token_and_state():
    assert name_first_token("acme forge") == "acme"
    assert name_first_token("") == ""
    assert state_code(" il ") == "IL"
    assert state_code(None) == ""
