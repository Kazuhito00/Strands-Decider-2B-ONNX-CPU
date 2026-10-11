# -*- coding: utf-8 -*-
"""Gate: what sits in the padding cannot change the hidden states at the read positions (causal model, positions precede it)."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strands_decide.runtime import Decider  # noqa: E402


def main() -> int:
    golden = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))
    d = Decider(str(ROOT / "models" / "onnx"))
    rng = np.random.default_rng(7)
    worst = 0.0
    for case, qid in (("choice-basic", "team"), ("multi", "urgency"), ("long-state", "team")):
        q = golden["cases"][case]["questions"][qid]
        ids = q["state_ids"] + q["question_ids"]
        n = len(ids)
        L = next(x for x in d.lengths if x >= n)
        sess = d._session(L)
        base = d.hidden(ids, q["read_positions"])
        for label, fill in (("random tokens", rng.integers(0, 200000, size=L - n)), ("zeros", np.zeros(L - n, dtype=np.int64))):
            input_ids = np.zeros((1, L), dtype=np.int64)
            input_ids[0, :n] = ids
            input_ids[0, n:] = fill
            mask = np.zeros((1, L), dtype=np.int64)
            mask[0, :n] = 1
            feeds = {"input_ids": input_ids, "attention_mask": mask,
                     "position_ids": np.broadcast_to(np.arange(L, dtype=np.int64), (3, 1, L)).copy(),
                     "read_positions": np.asarray(q["read_positions"], dtype=np.int64),
                     "image_features": np.zeros((0, 2048), np.float32), "image_slots": np.zeros((0,), np.int64)}
            got = sess.run(["hidden"], feeds)[0]
            diff = float(np.abs(got - base).max())
            worst = max(worst, diff)
            print("%-14s %-4d tok  padding=%-13s max|dh| %.2e" % (case + "/" + qid, n, label, diff), flush=True)
    print("worst %.2e" % worst)
    return 1 if worst > 1e-3 else 0


if __name__ == "__main__":
    raise SystemExit(main())
