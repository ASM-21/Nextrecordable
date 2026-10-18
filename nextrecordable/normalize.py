"""Name, address and ZIP normalization: the single source of truth.

Every place that needs a normalized name or ZIP (the ITA panel key, the DOL
inspection key, the matcher, API lookups) calls these Python functions. SQL
never re-implements them; pipeline steps map distinct raw values through
these functions and join the results back (see `keys.py`). Two
implementations of "the same" rule drift apart, and when they drift the
soft-key joins silently return nothing.
"""

from __future__ import annotations

import math
import re

# Business entity suffixes stripped from the END of a name only, so
# "Inc Magnetics LLC" keeps its leading "inc".
ENTITY_SUFFIXES = frozenset(
    {
        "inc", "incorporated",
        "llc", "lc",
        "ltd", "limited",
        "corp", "corporation",
        "co", "company",
        "lp", "llp",
        "pllc", "pc", "plc",
        "gmbh", "ag",
    }
)

# Expansions applied to name tokens. Single-letter directions are deliberately
# absent: in names "S" is usually an initial ("S and S Machine"), not "south".
NAME_ABBREVIATIONS = {
    "mfg": "manufacturing",
    "mfr": "manufacturer",
    "mfrs": "manufacturers",
    "mfgs": "manufacturing",
    "intl": "international",
    "natl": "national",
    "assoc": "associates",
    "assn": "association",
    "svcs": "services",
    "svc": "service",
    "bros": "brothers",
    "dist": "distribution",
    "ctr": "center",
    "hosp": "hospital",
    "univ": "university",
}

ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "avenue": "ave",
    "av": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "road": "rd",
    "lane": "ln",
    "place": "pl",
    "court": "ct",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "suite": "ste",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
}

# Long suffixes also match when misspelled ("Corproation", "Incorperated"):
# typos there are common and the token carries no identifying information.
LONG_ENTITY_SUFFIXES = ("corporation", "incorporated", "company", "limited")

_APOSTROPHES_RE = re.compile("['`" + chr(0x2019) + chr(0x2018) + "]")  # straight and curly
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\D")
_NUMBER_TOKEN_RE = re.compile(r"\d+")


def _is_missing(value) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _base_clean(text: str) -> str:
    s = text.lower()
    s = s.replace("&", " and ")
    s = _APOSTROPHES_RE.sub("", s)  # "McDonald's" -> "mcdonalds", not "mcdonald s"
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _osa_distance(a: str, b: str) -> int:
    """Optimal string alignment distance (edits, with adjacent transpositions)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def _is_entity_suffix(token: str) -> bool:
    if token in ENTITY_SUFFIXES:
        return True
    return len(token) >= 6 and any(
        abs(len(token) - len(s)) <= 2 and _osa_distance(token, s) <= 2 for s in LONG_ENTITY_SUFFIXES
    )


def normalize_name(name) -> str:
    """Normalize an establishment or company name.

    Lowercase, '&' -> 'and', apostrophes dropped, other punctuation to spaces,
    abbreviations expanded, leading "the" and trailing entity suffixes removed.
    Idempotent: normalize_name(normalize_name(x)) == normalize_name(x).
    """
    if _is_missing(name):
        return ""
    tokens = _base_clean(str(name)).split()
    tokens = [NAME_ABBREVIATIONS.get(t, t) for t in tokens]
    while len(tokens) > 1 and tokens[0] == "the":
        tokens = tokens[1:]
    while len(tokens) > 1 and _is_entity_suffix(tokens[-1]):
        tokens.pop()
    return " ".join(tokens)


def normalize_address(addr) -> str:
    """Normalize a street address: lowercase, abbreviate street types and directions."""
    if _is_missing(addr):
        return ""
    tokens = _base_clean(str(addr)).split()
    return " ".join(ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens)


def zip5(value) -> str:
    """Canonical 5-digit ZIP, or "" when the value can't be a ZIP.

    Handles the ways real files mangle ZIPs:
      "12345", "12345-6789", "123456789"   -> "12345"
      6511 (an int that lost its zero)      -> "06511"
      "6511-1234" / 65111234                -> "06511"
      "6511.0" (a float written by Excel)   -> "06511"
    Leading zeros matter: CT, MA, ME, NH, NJ, PR, RI and VT ZIPs start with 0,
    and DuckDB types an all-digit ZIP column as an integer unless told not to.
    """
    if _is_missing(value):
        return ""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = _DIGITS_RE.sub("", s)
    n = len(digits)
    if n == 5:
        return digits
    if n == 9:
        return digits[:5]
    if n in (3, 4):
        return digits.zfill(5)
    if n in (6, 7, 8):
        return digits.zfill(9)[:5]
    return ""


def name_first_token(normalized_name: str) -> str:
    """First token of an already-normalized name (a matcher blocking key)."""
    return normalized_name.split(" ", 1)[0] if normalized_name else ""


def name_numbers(normalized_name: str) -> str:
    """Numeric tokens in a name, canonicalized and sorted: "store 0123 plant 2" -> "123 2".

    Store and plant numbers are the main thing separating "Acme Plant 2" from
    "Acme Plant 3" at the same ZIP; string similarity alone scores them ~0.98.
    """
    if not normalized_name:
        return ""
    nums = sorted({str(int(t)) for t in _NUMBER_TOKEN_RE.findall(normalized_name)})
    return " ".join(nums)


def state_code(value) -> str:
    """Two-letter state/territory code, uppercased; "" when missing."""
    if _is_missing(value):
        return ""
    return str(value).strip().upper()[:2]
