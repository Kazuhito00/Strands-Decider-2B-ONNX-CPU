# -*- coding: utf-8 -*-
"""Run Strands' own engine (strands_decider, torch, CPU) on export/cases.py and write artifacts/golden.json (+ .npz).

    .venv-ref\\Scripts\\python.exe export/make_golden.py

For every question of every case it records what the ONNX port must reproduce: the token ids of the prompt (state and
question tokenized separately, as the engine does), the read positions (the <answer> token, then the last token of each
option line), the torso's hidden states at those positions (golden_hidden.npz), the logits before and after the
temperature, and the probabilities. Plain files only (JSON + npz), no pickles.

The engine is run the way the ONNX graph runs: one full prompt per question (``use_prefix_cache=False``), torso in fp32
on CPU (the engine's own CPU choice). ``models/oracle`` is a copy of the release whose config points the base model at
the local download instead of the Hub.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.cases import CASES  # noqa: E402

RELEASE = ROOT / "models" / "release"
ORACLE = ROOT / "models" / "oracle"
BASE = ROOT / "models" / "base"
OUT = ROOT / "artifacts"


def make_oracle_dir() -> pathlib.Path:
    """The release, with the base model pointed at the local copy (nothing is downloaded)."""
    if ORACLE.exists():
        shutil.rmtree(ORACLE)
    shutil.copytree(RELEASE, ORACLE, ignore=shutil.ignore_patterns("README.md", "LICENSE.md", "chat_template.jinja"))
    cfg_path = ORACLE / "strands_decider_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model"] = str(BASE)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return ORACLE


def main() -> int:
    from strands_decider.infer import EngineConfig, SystemOneEngine, _option_token_index, _to_answer
    from strands_decider.modeling import StrandsDeciderModel, apply_temperature, gather_options, pool_last_token
    from strands_decider.prompting import render_question, render_state
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

    path = make_oracle_dir()
    started = time.time()
    with torch.inference_mode():
        model = StrandsDeciderModel.load(str(path))
        engine = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=False))
    cfg = engine.model.config
    print("engine ready in %.0f s; torso dtype %s; temperature %.4f %s" % (
        time.time() - started, next(engine.model.torso.parameters()).dtype, cfg.temperature, cfg.temperature_by_kind),
        flush=True)

    kinds = {"noul": NoulQuestion, "choice": ChoiceQuestion, "score": ScoreQuestion}
    golden = {"source": {"release": json.loads((RELEASE / "provenance.json").read_text(encoding="utf-8"))["name"],
                         "base_model_revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c", "device": "cpu",
                         "torso_dtype": "float32", "torch": torch.__version__,
                         "max_length": cfg.max_length, "temperature_by_kind": cfg.temperature_by_kind},
              "cases": {}}
    hidden_store = {}
    for name, request in CASES.items():
        t0 = time.time()
        state_text = render_state(request["state"])
        case = {"request": request, "state_text": state_text, "questions": {}}
        for qid, spec in request["questions"].items():
            question = kinds[spec["type"]](**spec)
            rq = render_question(question)
            with torch.inference_mode():
                s, q = engine._fit(state_text, [rq.text])
                offsets = engine._last_offsets[0]
                ids, mask = engine._pad([s + q[0]])
                opt = engine._option_idx([rq], len(s))
                hidden = engine.model.encode(ids, mask)
                pooled = pool_last_token(hidden, mask).to(torch.float32)
                options = gather_options(hidden, opt).to(torch.float32)
                raw = engine.model.head(pooled, options)
                temp = engine._temperatures([rq.kind])
                logits = apply_temperature(raw, temp)
                probs = torch.softmax(logits[0, : rq.n_slots].double(), dim=-1).tolist()
            n_state = len(s)
            answer_pos = int(mask.sum().item()) - 1
            read_positions = [answer_pos] + [int(i) for i in opt[0, : rq.n_slots].tolist()]
            key = "%s/%s" % (name, qid)
            hidden_store[key] = torch.cat([pooled, options[0, : rq.n_slots]], dim=0).numpy()   # [1 + K, 2048], torso output
            case["questions"][qid] = {
                "kind": rq.kind, "question_text": rq.text, "slot_labels": list(rq.slot_labels),
                "state_ids": s, "question_ids": q[0], "state_len": n_state,
                "option_token_index": [int(i) for i in opt[0, : rq.n_slots].tolist()],
                "read_positions": read_positions,
                "raw_logits": raw[0, : rq.n_slots].tolist(), "temperature": float(temp[0]),
                "logits": logits[0, : rq.n_slots].tolist(), "probabilities": probs,
                "answer": _to_answer(rq, probs, ordinal_smoothing=cfg.ordinal_smoothing).model_dump(),
            }
        golden["cases"][name] = case
        top = ["%s:%.3f" % (qid, max(c["probabilities"])) for qid, c in case["questions"].items()]
        print("%-16s %5.1f s  %s" % (name, time.time() - t0, " ".join(top)), flush=True)
        OUT.mkdir(exist_ok=True)
        (OUT / "golden.json").write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")
        np.savez(OUT / "golden_hidden.npz", **{k.replace("/", "__"): v for k, v in hidden_store.items()})
    print("wrote", OUT / "golden.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
