# -*- coding: utf-8 -*-
"""Run Strands' own VISION engine (torch, CPU, fp32) on export/cases_image.py and write artifacts/golden_image.json (+ .npz).

    .venv-ref\\Scripts\\python.exe export/make_golden_image.py

strands_decider 0.1.0 (PyPI) has no image path; it is on the code repository's main, cloned to models/upstream by
`git clone --depth 1 https://github.com/strands-labs/strands-decider models/upstream` (the commit is recorded in the golden file). For every
question this records the prompt ids, the read positions, the torso hidden states at them, the probabilities, and per
image the vision tower's output, so the ONNX port can be compared stage by stage.
"""
from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "upstream" / "src"))      # main branch, ahead of the installed 0.1.0
from export.cases_image import CASES, images  # noqa: E402

ORACLE = ROOT / "models" / "oracle"
OUT = ROOT / "artifacts"


def main() -> int:
    import strands_decider
    from strands_decider.infer import EngineConfig, _to_answer
    from strands_decider.modeling import apply_temperature  # noqa: F401
    from strands_decider.prompting import render_question
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
    from strands_decider.vision import (VisionDeciderModel, VisionEngine, VisionEngineConfig, decode_image, fit_image,
                                        qwen_base)

    if not hasattr(VisionEngine, "_image_probs"):
        raise SystemExit("strands_decider on the path has no vision engine (%s)" % strands_decider.__file__)
    commit = subprocess.run(["git", "-C", str(ROOT / "models" / "upstream"), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    started = time.time()
    with torch.inference_mode():
        model = VisionDeciderModel.load(str(ORACLE))
        engine = VisionEngine(model, VisionEngineConfig(device="cpu", use_prefix_cache=False))
    first = next(engine.model.torso.parameters())
    print("vision engine ready in %.0f s; dtype %s; upstream %s" % (time.time() - started, first.dtype, commit[:10]), flush=True)

    captured = []
    engine.model.head.register_forward_hook(lambda m, args, out: captured.append([a.detach().clone() for a in args]))
    kinds = {"noul": NoulQuestion, "choice": ChoiceQuestion, "score": ScoreQuestion}
    pics = images()
    b64 = {k: base64.b64encode(v).decode() for k, v in pics.items()}
    golden = {"source": {"upstream_commit": commit, "device": "cpu", "torso_dtype": str(first.dtype), "torch": torch.__version__,
                         "image_long_side": 448, "max_length": engine.model.config.max_length},
              "cases": {}, "features": {}}
    store = {}
    base = qwen_base(engine.model.torso)
    for name, bytes_ in pics.items():
        img = fit_image(decode_image(b64[name]), 448)
        proc = engine.image_processor(images=[img], return_tensors="pt")
        with torch.inference_mode():
            feats = base.visual(proc["pixel_values"], grid_thw=proc["image_grid_thw"], return_dict=True).pooler_output
        store["features__" + name] = feats.float().numpy()
        golden["features"][name] = {"grid_thw": proc["image_grid_thw"].tolist(), "tokens": int(feats.shape[0])}
        print("  image %-10s grid %s -> %d tokens" % (name, proc["image_grid_thw"].tolist(), feats.shape[0]), flush=True)

    for cname, request in CASES.items():
        t0 = time.time()
        case = {"state": request["state"], "images": request["images"], "questions": {}}
        for qid, spec in request["questions"].items():
            question = kinds[spec["type"]](**spec)
            rq = render_question(question)
            captured.clear()
            with torch.inference_mode():
                imgs = [fit_image(decode_image(b64[n]), 448) for n in request["images"]]
                s, q, mm = engine._fit_images(request["state"], imgs, [rq.text])
                probs, ntok = engine._image_probs(request["state"], imgs, [rq])
            pooled, options = captured[-1]
            opt = engine._option_idx([rq], len(s))
            read_positions = [len(s) + len(q[0]) - 1] + [int(i) for i in opt[0, : rq.n_slots].tolist()]
            p = probs[0, : rq.n_slots].double().tolist()
            store["%s__%s" % (cname, qid)] = torch.cat([pooled, options[0, : rq.n_slots]], dim=0).float().numpy()
            case["questions"][qid] = {
                "kind": rq.kind, "question_text": rq.text, "slot_labels": list(rq.slot_labels),
                "state_ids": s, "question_ids": q[0], "image_grid_thw": mm["image_grid_thw"].tolist(),
                "read_positions": read_positions, "probabilities": p,
                "answer": _to_answer(rq, p, ordinal_smoothing=engine.model.config.ordinal_smoothing).model_dump()}
        golden["cases"][cname] = case
        top = ["%s:%.3f" % (qid, max(c["probabilities"])) for qid, c in case["questions"].items()]
        print("%-18s %5.1f s  %s" % (cname, time.time() - t0, " ".join(top)), flush=True)
        OUT.mkdir(exist_ok=True)
        (OUT / "golden_image.json").write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")
        np.savez(OUT / "golden_image.npz", **store)
    print("wrote", OUT / "golden_image.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
