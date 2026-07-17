"""Name and address normalization for entity resolution.

This module produces normalized strings suitable as blocking keys and
similarity-comparison inputs for splink (or any fuzzy matcher). The goal
is high recall on true matches without inflating the candidate pair count.

Design choices documented inline.
"""

from __future__ import annotations

import re

# Business entity suffixes to strip from the END of a name (kept lowercase).
# We don't strip them mid-name (e.g. "Inc Magnetics LLC" stays as-is mid-name).
ENTITY_SUFFIXES = frozenset(
    {
        "inc", "incorporated",
        "llc", "lc", "l.l.c",
        "ltd", "limited",
        "corp", "corporation",
        "co", "company",
        "lp", "llp", "l.p", "l.l.p",
        "pllc", "pc", "pa", "plc",
        "gmbh", "ag", "sa",  # foreign forms, occasionally appear
    }
)

# Common business name abbreviations. Bidirectional expansion makes
# "ABC MFG" match "ABC Manufacturing".
NAME_ABBREVIATIONS = {
    "mfg": "manufacturing",
    "mfr": "manufacturer",
    "mfrs": "manufacturers",
    "intl": "international",
    "natl": "national",
    "assoc": "associates",
    "assn": "association",
    "svcs": "services",
    "svc": "service",
    "bros": "brothers",
    "&": "and",
    "n": "north",  # only when standalone token in name context
    "s": "south",
    "e": "east",
    "w": "west",
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


_PUNCT_RE = re.compile(r"[^\w\s&]")
_WS_RE = re.compile(r"\s+")


def _collapse_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def normalize_name(name: str | None) -> str:
    """Normalize an establishment / company name.

    Lowercase, drop punctuation, expand abbreviations, strip trailing
    entity suffixes (Inc, LLC, etc).
    """
    if not name:
        return ""
    s = name.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _collapse_ws(s)

    tokens = s.split()
    tokens = [NAME_ABBREVIATIONS.get(t, t) for t in tokens]

    # Strip entity suffixes from the END (may have stacked: "ABC Mfg Co Inc")
    while tokens and tokens[-1] in ENTITY_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def normalize_address(addr: str | None) -> str:
    """Normalize a street address: lowercase, abbreviate, dedupe whitespace."""
    if not addr:
        return ""
    s = addr.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _collapse_ws(s)
    tokens = s.split()
    tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(tokens)


def zip5(zip_value: str | int | None) -> str:
    """Return the leading 5 digits of a ZIP code, or empty string."""
    if zip_value is None:
        return ""
    s = str(zip_value).strip()
    # Strip any non-digit, then take first 5
    digits = re.sub(r"\D", "", s)
    return digits[:5] if len(digits) >= 5 else ""


def name_first_token(normalized_name: str) -> str:
    """The first token of a normalized name. Useful as a blocking key."""
    parts = normalized_name.split(" ", 1)
    return parts[0] if parts else ""
