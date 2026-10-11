# -*- coding: utf-8 -*-
"""Gate: the ONNX Decider (onnxruntime + numpy only) against the engine's golden probabilities, question by question."""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strands_decide.protocol import render_question, render_state  # noqa: E402
from strands_decide.runtime import Decider  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=384, help="smallest torso graph (384/512/1024/1280)")
    args = ap.parse_args()
    golden = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))
    hid = np.load(ROOT / "artifacts" / "golden_hidden.npz")
    d = Decider(str(ROOT / "models" / "onnx"), seq_len=args.seq_len)
    worst_p = worst_h = 0.0
    bad = total = 0
    for name, case in golden["cases"].items():
        state_text = render_state(case["request"]["state"])
        for qid, want in case["questions"].items():
            rq = render_question(case["request"]["questions"][qid])
            probs, n = d.probabilities(state_text, rq)
            dp = float(np.abs(np.array(probs) - np.array(want["probabilities"])).max())
            top_ok = int(np.argmax(probs)) == int(np.argmax(want["probabilities"]))
            worst_p = max(worst_p, dp)
            total += 1
            bad += (not top_ok) or dp > 2e-2
            print("%-16s %-8s %4d tok  max|dp| %.2e  top %s" % (name, qid, n, dp, "ok" if top_ok else "DIFFERS"), flush=True)
    print("\nworst |dp| %.2e; failures %d of %d" % (worst_p, bad, total))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
