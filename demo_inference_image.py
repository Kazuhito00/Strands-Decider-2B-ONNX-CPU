# -*- coding: utf-8 -*-
"""Typed decisions about pictures with Strands Decider 2B, on onnxruntime + numpy + Pillow.

    python demo_inference_image.py                        # built-in example: artifacts/images/sample_chart.png (a bar chart)
    python demo_inference_image.py photo.jpg --noul "is=There is a dog in the picture"
    python demo_inference_image.py a.png b.png --state "Checkout page." --choice "next=retry,back,none"

Up to 4 images (PNG, JPEG, WebP, GIF); each is scaled down so its longer side is at most 448 px (196 tokens for a
square). The images go inside <state>; the model reads them with Qwen3.5's own vision tower. Needs vision_encoder.onnx in
the model directory (download_model.py). The torso graph must be at least as long as the prompt: when the default L=384
is too short, the runtime loads the next larger bucket that exists.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from demo_inference_text import parse_questions, show  # noqa: E402
from strands_decide import Decider, choice, add_device_args, providers_from_args, noul  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
SAMPLE = ROOT / "artifacts" / "images" / "sample_chart.png"   # syn-0001 of the Wikipedia image benchmark's synthetic set: drawn by a script


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files; omit to run the built-in example")
    ap.add_argument("--state", default="")
    ap.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    ap.add_argument("--score", action="append", metavar="NAME=low,...,high")
    ap.add_argument("--noul", action="append", metavar="NAME=statement")
    ap.add_argument("--instructions", default="")
    ap.add_argument("--model", default="models/onnx")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=384,
                    help="smallest torso graph (384/512/1024/1280); a longer request loads a larger one")
    add_device_args(ap)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    images, state, questions = args.images, args.state, parse_questions(args)
    if not images:
        if questions:
            ap.error("give an image, or omit the questions to run the built-in example")
        images, state = [str(SAMPLE)], "Look at the attached image."
        questions = {"tallest": choice("Which bar is the tallest?", {"A": "bar A", "B": "bar B", "C": "bar C", "D": "bar D"}),
                     "chart": noul("Is this picture a bar chart?")}
    elif not questions:
        ap.error("give at least one --choice, --score or --noul")

    providers, require_gpu = providers_from_args(args)
    decider = Decider(ROOT / args.model, intra_op_threads=args.threads, providers=providers, require_gpu=require_gpu,
                     seq_len=args.seq_len)
    started = time.perf_counter()
    answers = decider.decide(state, questions, images=images)
    elapsed = time.perf_counter() - started
    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return 0
    print("images: %s\n" % ", ".join(images))
    for name, answer in answers.items():
        show(name, answer)
    print("\ninference %.2f s (vision tower + %d questions, providers: %s)" % (elapsed, len(answers), ", ".join(decider.providers)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
