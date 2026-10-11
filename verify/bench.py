# -*- coding: utf-8 -*-
"""One measurement per process: latency and memory of one request shape.

    python verify/bench.py [--threads 8] [--runs 5] [--repeat 1] [--questions 3] [--options 3]

* "cold" is the first forward after load: the weights are paged in from disk, so it is far slower than the rest.
* "median" is over the following runs (the first of which is dropped as a warm-up).
* Peak memory is the process's peak working set.

Time depends on the machine and on what else is running: measure on a quiet machine, and do not run anything else while
this runs (the model needs ~8 GB of RAM).
"""
import argparse
import ctypes
import pathlib
import statistics
import sys
import time
from ctypes import wintypes

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strands_decide import Decider, choice, add_device_args, providers_from_args, noul, score  # noqa: E402

FILLER = ("The customer opened a ticket about the export button. It crashes in Safari but works in Chrome. "
          "Support replied within an hour and suggested switching browsers while engineering investigates. ")


def peak_mb():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    query = k32.K32GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    return round(c.PeakWorkingSetSize / 2**20, 0) if query(k32.GetCurrentProcess(), ctypes.byref(c), c.cb) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/onnx")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=384,
                    help="smallest torso graph (384/512/1024/1280); a longer request loads a larger one")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=1, help="how many times the state text is repeated")
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--options", type=int, default=3)
    add_device_args(ap)
    args = ap.parse_args()

    providers, require_gpu = providers_from_args(args)
    t0 = time.perf_counter()
    d = Decider(ROOT / args.model, intra_op_threads=args.threads,
                providers=providers, require_gpu=require_gpu, seq_len=args.seq_len)
    load = time.perf_counter() - t0
    kinds = [lambda i: choice("Which team should handle this (%d)?" % i,
                              {"opt%d" % k: "option number %d" % k for k in range(args.options)}),
             lambda i: noul("The customer is angry (%d)" % i),
             lambda i: score("How urgent is this (%d)?" % i, ["level %d" % k for k in range(args.options)])]
    questions = {"q%d" % i: kinds[i % 3](i) for i in range(args.questions)}
    request = {"state": FILLER * args.repeat, "questions": questions}
    from strands_decide.protocol import render_question, render_state
    state_text = render_state(request["state"])
    tokens = sum(d.probabilities(state_text, render_question(v.spec()))[1] for v in questions.values()) // len(questions)

    t = time.perf_counter()
    d.predict(request)
    cold = time.perf_counter() - t
    d.predict(request)  # warm-up
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        d.predict(request)
        times.append((time.perf_counter() - t) * 1000)
    print("tokens/question %d  questions %d  threads %d" % (tokens, args.questions, args.threads))
    provider = d.provider
    print("provider %s (session providers: %s)" % (provider, ", ".join(d.providers)))
    print("load %.1f s   cold first forward %.1f s" % (load, cold))
    print("median %.0f ms   min %.0f   max %.0f   (%d runs)   peak memory %s MB"
          % (statistics.median(times), min(times), max(times), args.runs, peak_mb()))


if __name__ == "__main__":
    main()
