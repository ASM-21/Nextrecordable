"""Tests for normalization functions."""

from osha_risk.normalize import (
    name_first_token,
    normalize_address,
    normalize_name,
    zip5,
)


class TestNormalizeName:
    def test_strips_entity_suffix(self):
        assert normalize_name("ABC Manufacturing Inc") == "abc manufacturing"
        assert normalize_name("ABC Manufacturing, LLC") == "abc manufacturing"
        assert normalize_name("A.B.C. Manufacturing Incorporated") == "a b c manufacturing"

    def test_handles_stacked_suffixes(self):
        # "ABC Mfg Co Inc" -> all three should strip
        assert normalize_name("ABC Mfg Co Inc") == "abc manufacturing"

    def test_expands_mfg_abbreviation(self):
        assert normalize_name("ABC MFG") == "abc manufacturing"
        assert normalize_name("ABC Mfg") == "abc manufacturing"

    def test_empty_and_none(self):
        assert normalize_name(None) == ""
        assert normalize_name("") == ""

    def test_ampersand_normalized(self):
        # & gets normalized to "and"
        result = normalize_name("Smith & Jones LLC")
        assert result == "smith and jones"

    def test_does_not_strip_middle_inc(self):
        # "Inc" mid-name should NOT be stripped (rare but possible)
        # Our rule: strip from END only.
        result = normalize_name("Acme Inc Holdings")
        assert "holdings" in result
        assert result.startswith("acme")

    def test_idempotent(self):
        s = "ABC Manufacturing, Inc."
        once = normalize_name(s)
        twice = normalize_name(once)
        assert once == twice


class TestNormalizeAddress:
    def test_abbreviates_street_types(self):
        assert normalize_address("123 Main Street") == "123 main st"
        assert normalize_address("456 Oak Avenue") == "456 oak ave"
        assert normalize_address("789 Industrial Boulevard") == "789 industrial blvd"

    def test_directions(self):
        assert normalize_address("100 North Main St") == "100 n main st"
        assert normalize_address("100 Northwest Pkwy") == "100 nw pkwy"

    def test_suite_normalized(self):
        assert normalize_address("123 Main St Suite 100") == "123 main st ste 100"

    def test_empty_and_none(self):
        assert normalize_address(None) == ""
        assert normalize_address("") == ""

    def test_already_abbreviated_unchanged(self):
        assert normalize_address("123 Main St Ste 100") == "123 main st ste 100"

    def test_punctuation_stripped(self):
        assert normalize_address("123 Main St., Suite #100") == "123 main st ste 100"


class TestZip5:
    def test_strips_plus_four(self):
        assert zip5("12345-6789") == "12345"
        assert zip5("12345") == "12345"

    def test_handles_int_input(self):
        assert zip5(12345) == "12345"

    def test_zero_padding_not_added(self):
        # If only 4 digits given, we return empty (invalid ZIP)
        assert zip5("1234") == ""

    def test_handles_none(self):
        assert zip5(None) == ""

    def test_handles_garbage(self):
        assert zip5("ZIP: 12345-6789!") == "12345"


class TestNameFirstToken:
    def test_extracts_first_token(self):
        normed = normalize_name("ABC Manufacturing Inc")
        assert name_first_token(normed) == "abc"

    def test_empty(self):
        assert name_first_token("") == ""
