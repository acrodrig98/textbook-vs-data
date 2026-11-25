"""Checks on the Week 1 prompt generator. These run anywhere, with no model."""
import re
from collections import Counter, defaultdict
from fractions import Fraction

import pytest

from prompts.generate import LAWS, PRIOR_LEVELS, law_value, make_items


@pytest.fixture(scope="module")
def items():
    return make_items(list(LAWS), [1, 2, 3, 4], ["mc", "numeric"], samples_per_cell=3, n_shots=0, seed=0)


def test_deterministic():
    a = make_items(["ohm"], [2], ["mc", "numeric"], 2, 0, seed=1)
    b = make_items(["ohm"], [2], ["mc", "numeric"], 2, 0, seed=1)
    assert a == b


def test_ids_unique(items):
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))


def test_rows_follow_data_law_and_exclude_x1(items):
    for it in items:
        for x, y in it["rows"]:
            assert x != 1
            assert y == law_value(it["k"], x, Fraction(it["p_data"]).limit_denominator(4))


def test_conflict_vs_congruent_labels(items):
    for it in items:
        if it["congruent"]:
            assert it["p_data"] == it["p_textbook"] and it["alt_kind"] == "distractor"
        else:
            assert it["p_alt"] == it["p_textbook"] and it["alt_kind"] == "textbook"
            assert it["shift"] > 0
        assert it["p_data"] != it["p_alt"]
        assert it["formula_data"] != it["formula_alt"]


def test_same_data_across_prior_levels(items):
    by_base = defaultdict(list)
    for it in items:
        by_base[(it["base_id"], it["format"], it["order"])].append(it)
    for group in by_base.values():
        assert sorted(it["prior_level"] for it in group) == sorted(PRIOR_LEVELS)
        assert len({str(it["rows"]) for it in group}) == 1
        assert len({it["target_data"] for it in group}) == 1


def test_mc_letters_balanced_and_valid(items):
    mc = [it for it in items if it["format"] == "mc"]
    assert Counter(it["target_data"] for it in mc)[" A"] == len(mc) / 2
    for it in mc:
        assert {it["target_data"], it["target_alt"]} == {" A", " B"}
        assert it["prompt"].endswith("\nAnswer:")
        letter = it["target_data"].strip()
        shown = re.search(rf"{letter}\) (.+?)(  B\)|\n)", it["prompt"]).group(1)
        assert shown == it["formula_data"]


def test_numeric_first_digits_differ_and_query_held_out(items):
    num = [it for it in items if it["format"] == "numeric"]
    assert num, "no numeric items generated"
    for it in num:
        assert it["target_data"] != it["target_alt"]
        assert it["target_data"] == str(it["y_data"])[0] and it["target_alt"] == str(it["y_alt"])[0]
        assert it["query_x"] not in {x for x, _ in it["rows"]}
        assert it["prompt"].endswith("=") and not it["prompt"].endswith(" =")


def test_prior_level_wording(items):
    for it in items:
        p = it["prompt"]
        if it["prior_level"] == "anonymous":
            assert p.startswith("Here are measurements of y versus x:")
        elif it["prior_level"] == "names_only":
            assert "law" not in p.lower()


def test_every_law_has_conflict_and_congruent_items(items):
    kinds = {(it["law"], it["format"], it["congruent"]) for it in items}
    for law in LAWS:
        for fmt in ("mc", "numeric"):
            assert (law, fmt, True) in kinds and (law, fmt, False) in kinds, (law, fmt)


def test_shots_are_prepended():
    it = make_items(["ohm"], [2], ["mc"], 1, n_shots=2, seed=0)[0]
    assert it["prompt"].count("Which formula fits the data?") == 3


def test_no_repeated_tables_and_laws_balanced(items):
    tables = {it["base_id"]: it for it in items}
    seen = set()
    for it in tables.values():
        key = (it["law"], it["p_data"], it["n_rows"], it["k"], str(it["rows"]))
        assert key not in seen, key
        seen.add(key)
    per_law = Counter(it["law"] for it in tables.values())
    # Laws with small x-grids can run out of distinct subsets at large row counts, so allow some slack
    assert max(per_law.values()) / min(per_law.values()) < 1.5, per_law


def test_fake_named_differs_from_named_only_by_the_name():
    items = make_items(list(LAWS), [2], ["mc", "numeric"], 1, 0, seed=0,
                       levels=("named_law", "fake_named"))
    by_key = {(it["base_id"], it["format"], it["order"], it["prior_level"]): it for it in items}
    pairs = 0
    for (base, fmt, order, level), it in by_key.items():
        if level != "named_law":
            continue
        fake = by_key[(base, fmt, order, "fake_named")]
        law = LAWS[it["law"]]
        assert law.real_name in it["prompt"] and law.real_name not in fake["prompt"]
        assert fake["prompt"] == it["prompt"].replace(law.real_name, law.fake_name)
        assert fake["target_data"] == it["target_data"] and fake["rows"] == it["rows"]
        pairs += 1
    assert pairs > 0


def test_adding_a_level_leaves_the_data_tables_unchanged():
    """New-level prompts must line up with earlier results by base_id."""
    base = make_items(list(LAWS), [1, 2, 3, 4], ["mc", "numeric"], 3, 0, seed=0)
    extra = make_items(list(LAWS), [1, 2, 3, 4], ["mc", "numeric"], 3, 0, seed=0, levels=("fake_named",))
    key = lambda it: (it["base_id"], it["format"], it["order"])
    base_tables = {key(it): (it["rows"], it["target_data"], it["target_alt"], it.get("query_x"))
                   for it in base if it["prior_level"] == "named_law"}
    extra_tables = {key(it): (it["rows"], it["target_data"], it["target_alt"], it.get("query_x")) for it in extra}
    assert base_tables == extra_tables


def test_fake_names_are_not_real_law_names():
    real_surnames = {"Ohm", "Hooke", "Galileo", "Coulomb", "Kepler", "Newton"}
    for law in LAWS.values():
        assert law.fake_name != law.real_name
        assert not any(name in law.fake_name for name in real_surnames), law.fake_name
