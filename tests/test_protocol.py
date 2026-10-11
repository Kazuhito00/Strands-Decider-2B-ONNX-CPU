# -*- coding: utf-8 -*-
"""Prompt rendering, validation, option read positions and answer arithmetic (strands_decide.protocol)."""
import numpy as np
import pytest

from strands_decide.protocol import (PointerHead, answer_from_probs, choice, derive_confidence, derive_score_confidence,
                                     noul, option_token_indices, render_question, render_state, score, softmax, validate)


def test_render_state_text_and_json():
    assert render_state("  hello  ") == "<state>\nhello\n</state>\n"
    assert render_state({"a": 1, "b": [2]}) == '<state>\n{\n  "a": 1,\n  "b": [\n    2\n  ]\n}\n</state>\n'
    assert render_state("") == "<state>\n\n</state>\n"


def test_render_choice_layout_and_spans():
    rq = render_question(choice("Which team?", {"billing": "charges  and\nrefunds", "tech": ""}).spec())
    assert rq.text == ('<question type="choice">\nSelect exactly one option.\nWhich team?\n<options>\n'
                       '1. billing — charges and refunds\n2. tech\n</options>\n</question>\n<answer>')
    assert rq.kind == "choice" and rq.slot_labels == ("billing", "tech") and rq.n_slots == 2
    assert [rq.text[a:b] for a, b in rq.option_spans] == ["1. billing — charges and refunds", "2. tech"]


def test_render_noul_uses_false_true_slots():
    rq = render_question(noul("The customer is angry").spec())
    assert rq.slot_labels == ("false", "true")
    assert "1. false — the statement does not hold for this state" in rq.text
    assert rq.text.endswith("</options>\n</question>\n<answer>")


def test_render_score_levels_are_numbered_from_zero():
    rq = render_question(score("How urgent?", ["low", "high"]).spec())
    assert rq.slot_labels == ("0", "1") and rq.slot_descriptions == ("low", "high")
    assert "1. 0 — low\n2. 1 — high" in rq.text


@pytest.mark.parametrize("spec", [
    {"type": "other"},
    {"type": "choice", "criteria": {"only": "one"}},
    {"type": "score", "criteria": ["a"]},
    {"type": "score", "criteria": [str(i) for i in range(11)]},
    {"type": "noul", "criteria": {"maybe": "x"}},
])
def test_validate_rejects(spec):
    with pytest.raises(ValueError):
        validate(spec)


def test_validate_accepts_limits():
    validate({"type": "choice", "criteria": {str(i): "" for i in range(255)}})
    validate({"type": "score", "criteria": [str(i) for i in range(10)]})
    validate({"type": "noul", "criteria": {"true": "yes"}})


def test_option_token_indices_picks_last_token_inside_each_span():
    offsets = [(0, 2), (2, 5), (5, 6), (6, 9), (9, 12)]
    assert option_token_indices(offsets, [(0, 6), (6, 12)]) == [2, 4]
    assert option_token_indices([(0, 0), (0, 3)], [(0, 3)]) == [1]            # empty spans (special tokens) are skipped


def test_option_token_indices_reports_truncated_option():
    with pytest.raises(ValueError, match="no tokens left"):
        option_token_indices([(10, 12)], [(0, 5)])


def test_confidence_bounds():
    assert derive_confidence([1.0, 0.0]) == 1.0
    assert derive_confidence([0.5, 0.5]) == 0.0
    assert derive_confidence([1.0]) == 1.0
    assert derive_score_confidence([1.0, 0.0, 0.0], 0.1) == 1.0
    assert 0.0 <= derive_score_confidence([0.4, 0.2, 0.4], 0.1) < 0.5


def test_answer_from_probs_shapes():
    c = answer_from_probs(render_question(choice("q", {"a": "", "b": ""}).spec()), [0.2, 0.8])
    assert c["type"] == "choice" and c["choice"] == "b" and c["probabilities"] == {"a": 0.2, "b": 0.8}
    n = answer_from_probs(render_question(noul("q").spec()), [0.25, 0.75])
    assert n == {"type": "noul", "noul": 0.75}
    s = answer_from_probs(render_question(score("q", ["low", "mid", "high"]).spec()), [0.0, 0.5, 0.5])
    assert s["score"] == 1.5 and s["legend"] == {"0": "low", "1": "mid", "2": "high"}


def _head(rng, d=8, p=4):
    w = {"norm.weight": rng.normal(size=d), "norm.bias": rng.normal(size=d),
         "q.weight": rng.normal(size=(p, d)), "q.bias": rng.normal(size=p),
         "k.weight": rng.normal(size=(p, d)), "k.bias": rng.normal(size=p)}
    return PointerHead({k: v.astype(np.float32) for k, v in w.items()}), w


def test_pointer_head_matches_a_direct_computation():
    rng = np.random.default_rng(0)
    head, w = _head(rng)
    answer, options = rng.normal(size=8).astype(np.float32), rng.normal(size=(3, 8)).astype(np.float32)

    def norm(h):
        h = h.astype(np.float64)
        return (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-5) * w["norm.weight"] + w["norm.bias"]

    q = norm(answer) @ w["q.weight"].T + w["q.bias"]
    k = norm(options) @ w["k.weight"].T + w["k.bias"]
    np.testing.assert_allclose(head.logits(answer, options), (k @ q) / 2.0, rtol=1e-4, atol=1e-4)


def test_softmax_is_a_distribution_and_stable():
    p = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert abs(p.sum() - 1.0) < 1e-12 and p.argmax() == 1
