# -*- coding: utf-8 -*-
"""The pure-Python Qwen byte-level BPE: a tiny tokenizer.json always, the real one when models/onnx has it."""
import json
import pathlib

import pytest

from strands_decide.protocol import choice, option_token_indices, render_question, render_state
from strands_decide.runtime import MAX_QUESTION_FRACTION, fit_to_window
from strands_decide.tokenizer import QwenBPE

ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL = ROOT / "models" / "onnx" / "tokenizer.json"


def test_merges_follow_rank_order(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    assert tok.encode("the") == [tok.vocab["the"]]
    assert tok.encode("and") == [tok.vocab["and"]]
    assert tok.encode(" the") == [tok.vocab["Ġthe"]]


def test_bytes_outside_ascii_become_byte_tokens(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    assert len(tok.encode("é")) == 2                         # two UTF-8 bytes, no merge for them
    assert tok.encode("") == []


def test_added_tokens_are_cut_out_of_the_text(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    eot, pad = tok._added["<|endoftext|>"], tok._added["<|image_pad|>"]
    assert tok.encode("<|endoftext|>") == [eot]
    assert tok.encode("a<|image_pad|><|image_pad|>b") == [tok.vocab["a"], pad, pad, tok.vocab["b"]]


def test_offsets_cover_the_text(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    text = "the and <|endoftext|>x"
    ids, offsets = tok.encode_with_offsets(text)
    assert ids == tok.encode(text) and len(ids) == len(offsets)
    assert offsets[0] == (0, 3)
    assert all(a < b for a, b in offsets) and offsets == sorted(offsets)
    assert text[offsets[-2][0]:offsets[-2][1]] == "<|endoftext|>"


def test_unsupported_pipeline_is_refused(tiny_tokenizer_json, tmp_path):
    spec = json.loads(pathlib.Path(tiny_tokenizer_json).read_text(encoding="utf-8"))
    spec["normalizer"] = None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="NFC"):
        QwenBPE(str(bad))
    spec["normalizer"] = {"type": "NFC"}
    spec["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = True
    bad.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="ByteLevel"):
        QwenBPE(str(bad))


def test_fit_to_window_gives_the_question_first_claim(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    rq = render_question(choice("which?", {"a": "", "b": ""}).spec())
    s_ids, q_ids, q_offsets = fit_to_window(tok, 400, render_state("x" * 500), rq)
    assert len(q_ids) == len(q_offsets) and len(s_ids) + len(q_ids) <= 400
    assert q_ids == tok.encode(rq.text)                          # the question fits whole
    assert len(s_ids) == 400 - len(q_ids)                        # the state is cut from the end to what is left
    option_token_indices(q_offsets, rq.option_spans)             # and every option line still has tokens


def test_fit_to_window_front_truncates_a_huge_question(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    rq = render_question(choice("which?", {"opt%d" % i: "d" * 40 for i in range(40)}).spec())
    s_ids, q_ids, q_offsets = fit_to_window(tok, 200, render_state("s"), rq)
    assert len(q_ids) == int(200 * MAX_QUESTION_FRACTION)
    assert tok.encode(rq.text)[-len(q_ids):] == q_ids            # the tail (last options, <answer>) is what survives


@pytest.mark.skipif(not REAL.exists(), reason="models/onnx/tokenizer.json not downloaded (download_model.py)")
def test_golden_prompt_ids_with_the_real_tokenizer():
    golden = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))
    tok = QwenBPE(str(REAL))
    for case in golden["cases"].values():
        state_text = render_state(case["request"]["state"])
        for qid, want in case["questions"].items():
            rq = render_question(case["request"]["questions"][qid])
            s_ids, q_ids, offsets = fit_to_window(tok, golden["source"]["max_length"], state_text, rq)
            assert s_ids == want["state_ids"] and q_ids == want["question_ids"]
            reads = [len(s_ids) + len(q_ids) - 1] + [len(s_ids) + i for i in option_token_indices(offsets, rq.option_spans)]
            assert reads == want["read_positions"]
