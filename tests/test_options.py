# -*- coding: utf-8 -*-
"""Command-line option parsing of the demos (no model is loaded)."""
import argparse

import demo_inference_text as demo
from strands_decide import choice, noul, score
from strands_decide.protocol import render_question


def _args(**kw):
    base = dict(choice=None, score=None, noul=None, instructions="")
    base.update(kw)
    return argparse.Namespace(**base)


def test_parse_questions_builds_each_kind():
    q = demo.parse_questions(_args(choice=["team=billing, tech"], score=["urgency=low,high"],
                                   noul=["angry=The customer is angry"]))
    assert q["team"] == choice("team", {"billing": "billing", "tech": "tech"})
    assert q["urgency"] == score("urgency", ["low", "high"])
    assert q["angry"] == noul("The customer is angry")


def test_instructions_apply_to_choice_and_score_only():
    q = demo.parse_questions(_args(choice=["t=a,b"], noul=["n=statement"], instructions="Be strict."))
    assert q["t"].instructions == "Be strict." and q["n"].instructions == "statement"


def test_builtin_example_is_valid():
    for question in demo.EXAMPLE_QUESTIONS.values():
        render_question(question.spec())
