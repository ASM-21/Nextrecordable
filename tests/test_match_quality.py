"""Match-quality harness: accuracy floors per perturbation, false-match ceilings.

Each case perturbs real directory entries the way people and other datasets
do, looks them up one at a time (the shipped matcher trained on each request,
so single typo'd lookups were the weak spot: 12 of 40), and checks how many
come back as the right establishment and how many come back as a WRONG one.
"""

from __future__ import annotations

import random
import re

import pytest

from nextrecordable.keys import python_key
from nextrecordable.match import match_rows

SAMPLE = 200


def _typo(name: str, rng: random.Random) -> str:
    words = name.split()
    i = max(range(len(words)), key=lambda k: len(words[k]))
    w = words[i]
    j = rng.randrange(1, len(w) - 2) if len(w) > 3 else 0
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


CASES = {
    "exact": (lambda e, r: {"name": e["establishment_name"], "street_address": e["street_address"]}, 1.00),
    "uppercase_no_street": (lambda e, r: {"name": e["establishment_name"].upper()}, 1.00),
    "abbreviated": (lambda e, r: {"name": e["establishment_name"].replace("Manufacturing", "Mfg")
                                  .replace(" and ", " & ").replace("Corporation", "Corp.")}, 1.00),
    "one_typo": (lambda e, r: {"name": _typo(e["establishment_name"], r)}, 0.97),
    "one_typo_with_street": (lambda e, r: {"name": _typo(e["establishment_name"], r),
                                           "street_address": e["street_address"]}, 0.99),
    "zip_plus_4": (lambda e, r: {"name": e["establishment_name"], "zip_code": e["zip_code"] + "-4307"}, 1.00),
    "zip_lost_leading_zero": (lambda e, r: {"name": e["establishment_name"], "zip_code": str(int(e["zip_code"]))}, 1.00),
}


def _lookup(con, estabs, make, rng):
    rows = []
    for e in estabs:
        row = {"state": e["state"], "zip_code": e["zip_code"], **make(e, rng)}
        rows.append(row)
    res = match_rows(con, rows)
    truth = [python_key(e["establishment_name"], e["state"], e["zip_code"]) for e in estabs]
    correct = wrong = 0
    for (_, r), t in zip(res.iterrows(), truth, strict=True):
        if r["status"] == "matched":
            correct += r["estab_key"] == t
            wrong += r["estab_key"] != t
    return correct, wrong


@pytest.mark.parametrize("case", list(CASES))
def test_accuracy_floor_and_no_wrong_matches(world, case):
    make, floor = CASES[case]
    rng = random.Random(11)
    sample = rng.sample(world["estabs"], SAMPLE)
    # one establishment per lookup, as the API does
    correct = wrong = 0
    for e in sample[:40]:
        c, w = _lookup(world["con"].cursor(), [e], make, rng)
        correct += c
        wrong += w
    c, w = _lookup(world["con"].cursor(), sample[40:], make, rng)
    correct += c
    wrong += w
    assert wrong == 0
    assert correct / SAMPLE >= floor, f"{case}: {correct}/{SAMPLE}"


def test_different_plant_number_is_not_matched(world):
    plants = [e for e in world["estabs"] if re.search(r"Plant \d+", e["establishment_name"])]
    assert len(plants) > 50
    rows = [{"name": re.sub(r"Plant (\d+)", lambda m: f"Plant {int(m.group(1)) + 1}", e["establishment_name"]),
             "state": e["state"], "zip_code": e["zip_code"]} for e in plants]
    res = match_rows(world["con"].cursor(), rows)
    originals = {python_key(e["establishment_name"], e["state"], e["zip_code"]) for e in plants}
    false_matches = res[(res["status"] == "matched") & res["estab_key"].isin(originals)]
    assert len(false_matches) == 0


def test_similar_name_in_same_zip_is_not_matched(world):
    """'Summit Castings' vs 'Summit Coatings' (Jaro-Winkler 0.945) in one ZIP."""
    con = world["con"].cursor()
    e = next(x for x in world["estabs"] if "Castings" in x["establishment_name"])
    res = match_rows(con, [{"name": e["establishment_name"].replace("Castings", "Coatings"),
                            "state": e["state"], "zip_code": e["zip_code"]}])
    assert not (res.iloc[0]["status"] == "matched" and
                res.iloc[0]["estab_key"] == python_key(e["establishment_name"], e["state"], e["zip_code"]))


def test_unknown_companies_are_unmatched(world):
    rows = [{"name": f"Nonexistent Widgets {i}", "state": "IL", "zip_code": "60601"} for i in range(50)]
    res = match_rows(world["con"].cursor(), rows)
    assert set(res["status"]) == {"unmatched"}


def test_exact_name_in_wrong_zip_needs_the_street(world):
    con = world["con"].cursor()
    e = next(x for x in world["estabs"] if x["zip_code"][-1] != "9")  # +1 stays in the same ZIP3
    wrong_zip = f"{int(e['zip_code']) + 1:05d}"
    without = match_rows(con, [{"name": e["establishment_name"], "state": e["state"], "zip_code": wrong_zip}])
    assert without.iloc[0]["status"] != "matched" or without.iloc[0]["estab_key"] == python_key(
        e["establishment_name"], e["state"], e["zip_code"])
    with_street = match_rows(con, [{"name": e["establishment_name"], "state": e["state"], "zip_code": wrong_zip,
                                    "street_address": e["street_address"]}])
    assert with_street.iloc[0]["status"] == "matched"
    assert with_street.iloc[0]["estab_key"] == python_key(e["establishment_name"], e["state"], e["zip_code"])
