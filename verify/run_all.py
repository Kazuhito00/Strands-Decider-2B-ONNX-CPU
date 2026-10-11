# -*- coding: utf-8 -*-
"""Run every gate this repository has, in order, and summarise.

Gates that need only the runtime dependencies always run. Gates that need an oracle (tokenizers for the tokenizer fuzz,
torch + transformers for the vision inputs) are skipped with a reason rather than failing. Two gates load the 2B torso (~8 GB of RAM, a few minutes each);
``--skip-model`` leaves those out for a quick pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: (script, argv, modules it needs beyond the runtime, loads the 2B torso?, one-line purpose)
GATES = [
    ("check_prompt.py", [], [], False,
     "token ids and read positions vs the engine, exactly"),
    ("verify.py", [], [], True,
     "probabilities and decisions vs the engine (fp32)"),
    ("check_padding.py", [], [], True,
     "padding content cannot change the result"),
    ("check_tokenizer_fuzz.py", ["20000"], ["tokenizers"], False,
     "pure-Python tokenizer vs the Rust tokenizers (ids and offsets)"),
    ("check_vision_inputs.py", [], ["transformers", "torch"], False,
     "image pixels, bilinear and position inputs vs transformers (needs models/base)"),
    ("verify_image.py", [], ["PIL"], True,
     "image path (vision tower, prompt, probabilities) vs the vision engine"),
    ("check_combining.py", [], ["tokenizers"], False,
     "combining marks do not move option read positions"),
]

RUNTIME = {"onnxruntime": "inference", "numpy": "arrays"}
FORBIDDEN = ["torch", "transformers", "tokenizers", "jinja2", "onnx", "regex"]


def have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="interpreter to run the gates with")
    parser.add_argument("--skip-model", action="store_true", help="skip the gates that load the 2B torso")
    parser.add_argument("--full-fuzz", action="store_true", help="run the tokenizer fuzz at full size (slow)")
    args = parser.parse_args()

    print("%s\n" % ROOT.name)
    present = [m for m in RUNTIME if have(m)]
    missing = [m for m in RUNTIME if not have(m)]
    heavy = [m for m in FORBIDDEN if have(m)]
    print("  %-24s %-5s runtime: %s" % ("dependencies", "ok " if not missing else "XX ", ", ".join(present) or "none"))
    if missing:
        print("      missing runtime dependency: %s" % ", ".join(missing))
    if heavy:
        print("      note: %s present -- fine for development, not needed to run" % ", ".join(heavy))

    failed = skipped = passed = 0
    for script, argv, needs, loads_model, purpose in GATES:
        path = ROOT / "verify" / script
        if not path.exists():
            continue
        absent = [m for m in needs if not have(m)]
        if absent:
            print("  %-24s skip  needs %s -- %s" % (script[:-3], ", ".join(absent), purpose))
            skipped += 1
            continue
        if loads_model and args.skip_model:
            print("  %-24s skip  --skip-model -- %s" % (script[:-3], purpose))
            skipped += 1
            continue
        if script == "check_tokenizer_fuzz.py" and args.full_fuzz:
            argv = ["150000"]
        started = time.perf_counter()
        result = subprocess.run([args.python, str(path), *argv], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        ok = result.returncode == 0
        failed += not ok
        passed += ok
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        summary = lines[-1][:50] if lines else purpose
        print("  %-24s %-5s %-50s %6.1fs" % (script[:-3], "ok " if ok else "XX ", summary, elapsed))
        if not ok:
            for line in [l for l in (result.stdout + result.stderr).splitlines() if l.strip()][-6:]:
                print("      %s" % line[:110])
    print("\n%d passed, %d failed, %d skipped" % (passed, failed, skipped))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
