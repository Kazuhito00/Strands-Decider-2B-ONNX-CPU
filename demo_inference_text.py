# -*- coding: utf-8 -*-
"""Text decisions with Strands Decider 2B, on onnxruntime + numpy.

    python demo_inference_text.py
    python demo_inference_text.py --state "..." --choice "team=billing,technical"

One question is one forward pass of the 2B torso; the pointer head scores every option of that question. Nothing is
generated; each answer is one of the options supplied. Prompts longer than the exported graph length need a longer graph.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strands_decide import Decider, choice, add_device_args, providers_from_args, noul, score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

EXAMPLE_STATE = {
    "ticket": "My card was charged twice for the same order and support has not "
              "replied in 3 days.",
    "customer": {"tenure_months": 14, "vip": False},
}
EXAMPLE_QUESTIONS = {
    "angry": noul("The customer is angry"),
    "team": choice("Which team should handle this?",
                   {"billing": "charges and refunds", "tech": "bugs and outages",
                    "sales": "pricing and contracts"}),
    "urgency": score("How urgent is this?", ["low", "medium", "high", "critical"]),
}


def parse_questions(args):
    questions = {}
    for spec in args.choice or []:
        name, _, rest = spec.partition("=")
        options = [v.strip() for v in rest.split(",") if v.strip()]
        questions[name.strip()] = choice(args.instructions or name.strip(),
                                         {o: o for o in options})
    for spec in args.score or []:
        name, _, rest = spec.partition("=")
        levels = [v.strip() for v in rest.split(",") if v.strip()]
        questions[name.strip()] = score(args.instructions or name.strip(), levels)
    for spec in args.noul or []:
        name, _, rest = spec.partition("=")
        questions[name.strip()] = noul(rest.strip() or name.strip())
    return questions


def show(name: str, answer: dict) -> None:
    if answer["type"] == "noul":
        print("%-10s noul   p(yes)=%.3f" % (name, answer["noul"]))
        return
    spread = "  ".join(
        "%s=%.3f" % kv
        for kv in sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])
    )
    if answer["type"] == "choice":
        print("%-10s choice %-20s confidence %.3f\n           %s"
              % (name, answer["choice"], answer["confidence"], spread))
    else:
        legend = answer["legend"]
        best = max(answer["probabilities"], key=answer["probabilities"].get)
        print("%-10s score  %-20s expected %.2f, confidence %.3f\n           %s"
              % (name, legend[best], answer["score"], answer["confidence"], spread))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", help="text or JSON; omit to run the built-in example")
    parser.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    parser.add_argument("--score", action="append", metavar="NAME=low,...,high")
    parser.add_argument("--noul", action="append", metavar="NAME=statement")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--model", default="models/onnx")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=384,
                    help="smallest torso graph (384/512/1024/1280); a longer request loads a larger one")
    add_device_args(parser)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.state is None:
        state, questions = EXAMPLE_STATE, EXAMPLE_QUESTIONS
    else:
        try:
            state = json.loads(args.state)
        except json.JSONDecodeError:
            state = args.state
        questions = parse_questions(args)
        if not questions:
            parser.error("give at least one --choice, --score or --noul")

    started = time.perf_counter()
    providers, require_gpu = providers_from_args(args)
    decider = Decider(ROOT / args.model, intra_op_threads=args.threads, providers=providers, require_gpu=require_gpu,
                     seq_len=args.seq_len)
    load = time.perf_counter() - started

    started = time.perf_counter()
    answers = decider.decide(state, questions)
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return 0

    print("state:\n  %s\n" % json.dumps(state, ensure_ascii=False)[:160])
    for name, answer in answers.items():
        show(name, answer)
    print("\n%d questions, one forward pass each" % len(answers))
    print("load %.1f s, inference %.2f s (%s)" % (load, elapsed, ", ".join(decider.providers)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
