"""Checks on the scoring and analysis code, using synthetic scores (no model needed)."""
import numpy as np
import pandas as pd
import pytest

from eval.analyze import letter_bias, phase_cells, prior_pull, rates, table_decisions
from prompts.generate import LAWS, make_items

LETTER_BIAS = 5.0  # a strong made-up preference for " A"


def synthetic_results(pull_toward_textbook: bool) -> pd.DataFrame:
    """Fake scores: the model follows the data unless the law is named (if pull_toward_textbook),
    plus a large constant " A" preference that order-averaging must cancel out."""
    items = make_items(list(LAWS), [1, 2, 3], ["mc", "numeric"], samples_per_cell=2, n_shots=0, seed=0)
    rows = []
    for it in items:
        follows_data = it["congruent"] or not (pull_toward_textbook and it["prior_level"] == "named_law")
        diff = 2.0 if follows_data else -2.0
        if it["format"] == "mc":
            diff += LETTER_BIAS if it["target_data"] == " A" else -LETTER_BIAS
        rows.append({**it, "logit_data": diff / 2, "logit_alt": -diff / 2, "logit_diff": diff,
                     "p_two_targets": 0.9, "top1_token": it["target_data"]})
    return pd.DataFrame(rows)


def test_letter_bias_is_measured_then_cancelled():
    df = synthetic_results(pull_toward_textbook=True)
    assert letter_bias(df) == pytest.approx(LETTER_BIAS)
    dec = table_decisions(df)
    mc = dec[dec["format"] == "mc"]
    assert (mc["n_orders"] == 2).all()
    # despite the +5 letter bias (bigger than the +/-2 signal), every decision is still correct
    expected = mc["congruent"] | (mc["prior_level"] != "named_law")
    assert (mc["sides_with_data"] == expected).all()


def test_prior_pull_detected_when_present():
    dec = table_decisions(synthetic_results(pull_toward_textbook=True))
    for fmt in ("mc", "numeric"):
        pull, lo, hi, n = prior_pull(dec, fmt)
        assert pull == pytest.approx(1.0) and lo == pytest.approx(1.0) and n > 0


def test_no_prior_pull_when_absent():
    dec = table_decisions(synthetic_results(pull_toward_textbook=False))
    for fmt in ("mc", "numeric"):
        pull, lo, hi, _ = prior_pull(dec, fmt)
        assert pull == 0 and lo == 0 and hi == 0
    r = rates(dec)
    assert (r["rate"] == 1.0).all()


def test_phase_cells_cover_conflict_tables_only():
    dec = table_decisions(synthetic_results(pull_toward_textbook=True))
    cells = phase_cells(dec)
    assert cells["n_tables"].sum() == (~dec["congruent"]).sum()
    assert set(cells["shift"]) == {0.5, 1.0, 2.0}
    named = cells[cells["prior_level"] == "named_law"]["rate"]
    anon = cells[cells["prior_level"] == "anonymous"]["rate"]
    assert np.allclose(named, 0.0) and np.allclose(anon, 1.0)


def test_batched_scoring_matches_single_prompts():
    """Right padding must not change any prompt's last-token logits (needs a tiny model download)."""
    torch = pytest.importorskip("torch")
    from eval.behavior import load_model, next_token_logits
    try:
        model, tok = load_model("hf-internal-testing/tiny-random-gpt2", "float32", "cpu")
    except Exception as e:  # offline
        pytest.skip(f"tiny model unavailable: {e}")
    prompts = [it["prompt"] for it in make_items(["ohm", "kepler"], [1, 4], ["mc", "numeric"], 1, 0, 0)[:6]]
    batched = next_token_logits(model, tok, prompts, "cpu")
    single = torch.cat([next_token_logits(model, tok, [p], "cpu") for p in prompts])
    assert torch.allclose(batched, single, atol=1e-4)


def test_breakdown_with_made_up_law_level():
    """If only the real law name pulls, the whole pull shows up in the real -> made-up step."""
    from prompts.generate import ALL_LEVELS
    items = make_items(list(LAWS), [1, 2], ["mc"], samples_per_cell=2, n_shots=0, seed=0, levels=ALL_LEVELS)
    rows = []
    for it in items:
        follows = it["congruent"] or it["prior_level"] != "named_law"
        diff = 2.0 if follows else -2.0
        rows.append({**it, "logit_data": diff / 2, "logit_alt": -diff / 2, "logit_diff": diff,
                     "p_two_targets": 0.9, "top1_token": it["target_data"]})
    dec = table_decisions(pd.DataFrame(rows))
    real_step = prior_pull(dec, "mc", more_data="fake_named", less_data="named_law")
    framing_step = prior_pull(dec, "mc", more_data="names_only", less_data="fake_named")
    assert real_step[0] == pytest.approx(1.0) and framing_step[0] == pytest.approx(0.0)


def test_load_many_rejects_overlapping_files(tmp_path):
    import json
    from eval.analyze import load_many
    rec = {"id": "x-1", "meta": {"model": "m"}}
    for name in ("a.jsonl", "b.jsonl"):
        (tmp_path / name).write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="more than one file"):
        load_many([tmp_path / "a.jsonl", tmp_path / "b.jsonl"])
