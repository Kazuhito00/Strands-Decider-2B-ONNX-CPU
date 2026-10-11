# -*- coding: utf-8 -*-
"""Gate: the ONNX image path (Pillow + onnxruntime + numpy) against strands_decider's vision engine, stage by stage.

For every image: the vision tower's output vs the engine's (artifacts/golden_image.npz, stored for shapes / screen / chart only). For every question: our prompt ids
(state with image tokens, question) and read positions equal the engine's exactly, then the probabilities agree.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "export"))
from cases_image import CASES, images  # noqa: E402
from strands_decide.image import (VisionEncoder, expand_image_tokens, prepare_images, render_image_state)  # noqa: E402
from strands_decide.protocol import option_token_indices, render_question, render_state  # noqa: E402
from strands_decide.runtime import Decider  # noqa: E402


def main() -> int:
    golden = json.loads((ROOT / "artifacts" / "golden_image.json").read_text(encoding="utf-8"))
    npz = np.load(ROOT / "artifacts" / "golden_image.npz")
    pics = images()
    d = Decider(str(ROOT / "models" / "onnx"))
    enc = VisionEncoder(ROOT / "models" / "onnx", config=None)
    bad = 0

    print("vision tower")
    feats = {}
    needed = sorted({n for case in golden["cases"].values() for n in case["images"]} | set(golden["features"]))
    for name in needed:
        prepared = prepare_images([pics[name]], enc.config)
        f = enc.encode(prepared)
        feats[name] = (f, prepared)
        info = golden["features"].get(name)
        if info is None:          # the golden file keeps the vision output of a few images only (5 MB per file limit)
            print("  %-12s grid %-12s (no stored vision output; used by the question checks below)"
                  % (name, prepared.image_grid_thw[0].tolist()))
            continue
        want = npz["features__" + name]
        diff = float(np.abs(f - want).max())
        ok = list(prepared.image_grid_thw[0]) == info["grid_thw"][0] and f.shape == want.shape and diff < 5e-2
        bad += not ok
        print("  %-12s grid %-12s max|df| %.2e (scale %.1f)  %s" % (name, prepared.image_grid_thw[0].tolist(), diff,
                                                              float(np.abs(want).max()), "ok" if ok else "DIFFERS"))

    print("questions")
    worst = 0.0
    for cname, case in golden["cases"].items():
        prepared = prepare_images([pics[n] for n in case["images"]], enc.config)
        f = np.concatenate([feats[n][0] for n in case["images"]])
        state_text = expand_image_tokens(render_image_state(render_state(case["state"]), len(case["images"])),
                                         prepared.token_counts)
        for qid, want in case["questions"].items():
            rq = render_question(CASES[cname]["questions"][qid])
            s_ids, q_ids, offs = d.fit(state_text, rq)
            problems = []
            if rq.text != want["question_text"]:
                problems.append("question text")
            if s_ids != want["state_ids"]:
                problems.append("state ids")
            if q_ids != want["question_ids"]:
                problems.append("question ids")
            reads = [len(s_ids) + len(q_ids) - 1] + [len(s_ids) + i for i in option_token_indices(offs, rq.option_spans)]
            if reads != want["read_positions"]:
                problems.append("read positions")
            probs, n = d.probabilities(state_text, rq, image_features=f, image_grid_thw=prepared.image_grid_thw)
            dp = float(np.abs(np.array(probs) - np.array(want["probabilities"])).max())
            top = int(np.argmax(probs)) == int(np.argmax(want["probabilities"]))
            worst = max(worst, dp)
            if not top or dp > 3e-2:
                problems.append("probabilities")
            bad += bool(problems)
            print("  %-18s %-8s %4d tok  max|dp| %.2e  %s" % (cname, qid, n, dp, "ok" if not problems else "DIFFERS: " + ", ".join(problems)), flush=True)
    print("\nworst |dp| %.2e; failures %d" % (worst, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
