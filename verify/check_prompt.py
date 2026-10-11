# -*- coding: utf-8 -*-
"""Gate: our prompt text, token ids and read positions equal the engine's, question by question, exactly.

Needs only numpy (no model is loaded). The reference comes from artifacts/golden.json, which export/make_golden.py wrote
by running Strands' own code. Checks, for every question: the rendered question text, the state ids, the question ids,
the last token of every option line, and the read positions (the <answer> token, then those).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strands_decide.protocol import option_token_indices, render_question, render_state  # noqa: E402
from strands_decide.runtime import fit_to_window  # noqa: E402
from strands_decide.tokenizer import QwenBPE  # noqa: E402


def main() -> int:
    golden = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))
    tok = QwenBPE(str(ROOT / "models" / "onnx" / "tokenizer.json"))
    max_length = golden["source"]["max_length"]
    bad = total = 0
    for name, case in golden["cases"].items():
        state_text = render_state(case["request"]["state"])
        for qid, want in case["questions"].items():
            total += 1
            rq = render_question(case["request"]["questions"][qid])
            s_ids, q_ids, offs = fit_to_window(tok, max_length, state_text, rq)
            opts = option_token_indices(offs, rq.option_spans)
            problems = []
            if state_text != case["state_text"]:
                problems.append("state text")
            if rq.text != want["question_text"]:
                problems.append("question text")
            if s_ids != want["state_ids"]:
                problems.append("state ids")
            if q_ids != want["question_ids"]:
                problems.append("question ids")
            # the engine records them as positions in the whole prompt (state first), ours are question-relative
            if [len(s_ids) + i for i in opts] != want["option_token_index"]:
                problems.append("option positions")
            if [len(s_ids) + len(q_ids) - 1] + [len(s_ids) + i for i in opts] != want["read_positions"]:
                problems.append("read positions")
            if list(rq.slot_labels) != want["slot_labels"]:
                problems.append("slot labels")
            print("%-16s %-8s %4d tokens  %s" % (name, qid, len(s_ids) + len(q_ids),
                                                 "ok" if not problems else "DIFFERS: " + ", ".join(problems)))
            bad += bool(problems)
    print("\nfailures: %d of %d questions" % (bad, total))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
