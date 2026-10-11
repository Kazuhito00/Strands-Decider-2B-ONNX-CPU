[[Japanese](README.md)/[English](README_EN.md)]

# Strands-Decider-2B-ONNX-CPU

A minimal-dependency inference implementation of [`StrandsAgents/strands-decider-2B-hobson-v21`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) (Strands Decider, a structured decision model for text and images) that runs on onnxruntime and numpy only. GPU inference is possible too (but it was not faster on the GPU I tried; see below).<br>
It does not use torch or transformers, which the official inference code (`strands-decider`) requires.

# Features
- Minimal dependencies: only onnxruntime and numpy at run time (plus pillow when images are used)
- Two scripts are enough to run it: `download_model.py` (fetch the model) and `demo_inference_text.py` (inference). Images: `demo_inference_image.py`; camera: `demo_inference_webcam.py`
- Decisions match the reference implementation (the official Strands engine, PyTorch fp32): text 16/16, images 8/8 (see "Verification")
- Every question is answered by one forward pass (nothing is generated). Three question types: `noul` (yes/no), `choice` (one of 2 to 255 options), `score` (an ordered scale of 2 to 10 levels). Every answer carries a confidence
- Images (up to 4 per request; each image is scaled down to at most 448 px on the long side and embedded in `<state>`)
- Pre-processing (Qwen byte-level BPE, prompt layout, 3-axis M-RoPE, image patching and the vision-encoder inputs) is ported to pure Python, with verification scripts that compare it to the reference
- Runs on CPU, and on GPU (CUDA or DirectML) with `--gpu`

# Purpose of This Repository
- How far can Strands Decider 2B (Qwen3.5-2B-Base + LoRA + a pointer readout head) run with only 2 to 3 run-time dependencies?
- Can a pure-Python port of the reference pre-processing build the same inputs?
- Does the ONNX vision encoder give the same answers as the original implementation on image requests?

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   (only when images are used)
```
That is all that is needed to run. Fetching the model uses the standard library only.<br>
**About 8 GB of free memory or more is required** (the fp32 weights are 7.5 GB; the peak at sequence length L=384 is about 7.4 GB; images add the 1.3 GB vision encoder).<br>
Environments are built with [uv](https://docs.astral.sh/uv/). Optional dependency groups:
- `verify`: `tokenizers`, for the tokenizer oracle gates (`check_tokenizer_fuzz`, `check_combining`)
- `test`: pytest, for `tests/` (no model needed)
- `webcam`: OpenCV, for `demo_inference_webcam.py`
- `export`: torch, transformers and so on, only to export the ONNX yourself (never used at run time)

# Installation

```bash
git clone https://github.com/Kazuhito00/Strands-Decider-2B-ONNX-CPU
cd Strands-Decider-2B-ONNX-CPU

# only the run-time dependencies (onnxruntime, numpy, pillow)
uv sync
```
`uv sync` also installs this checkout's `strands_decide` package; `uv pip install git+https://github.com/Kazuhito00/Strands-Decider-2B-ONNX-CPU` gives you `import strands_decide` as well (the model files are fetched separately).

# Download Model
Strands Decider has no published ONNX. The exported ONNX files are meant to be attached to the [Releases](https://github.com/Kazuhito00/Strands-Decider-2B-ONNX-CPU/releases) of this repository (they are not in the repository itself; see `release/make_release.py` and "What the Release holds" below).<br>
This script fetches them into `models/onnx/`. It uses plain HTTPS from the standard library: no git, no git-lfs.
```bash
uv run download_model.py
```
- **With no arguments it fetches every file of the Release** (the graphs for L=384, 512 and 1280, the vision encoder, the head, the tokenizer and the licence texts: 18 assets, about 9.4 GB)
- By default it reads the latest Release of `Kazuhito00/Strands-Decider-2B-ONNX-CPU` (`--repo OWNER/NAME` or the environment variable `STRANDS_ONNX_REPO` selects another repository; precedence: argument > environment > default)
- Every file is checked against the SHA-256 in the Release's `manifest.json` (a mismatch deletes the file and stops)
- Files split to stay under GitHub's 2 GiB asset limit (the shared weights `torso_L512.onnx.data`, 7.5 GB, in 4 parts) are joined after download and then checked (while joining, about 7.5 GB of extra disk is used for a while)
- Files already in place are skipped, and an interrupted download resumes with the same command

```bash
uv run download_model.py --list                                          # list the files
uv run download_model.py --exclude "vision_encoder*"                     # text only (no vision encoder; about 1.3 GB less)
uv run download_model.py --seq-len 384                                   # only the default L=384 graph (the shared weights and the vision encoder come along)
uv run download_model.py --tag v0.0.0                                    # a given tag (default: latest)
```

The Release holds three graphs: L=384 (default), 512 and 1280. **The weights (`torso_L512.onnx.data`, 7.5 GB) are shared by all three, so an extra graph adds only about 0.17 GB.** For that reason `--seq-len 384` still fetches the 7.5 GB of weights (it saves only the two other graphs, about 0.35 GB). `manifest.json` lists the data files each `.onnx` requires, so `--seq-len` and `--exclude` can never leave a graph without its weights (even `--exclude "torso_L512*"` keeps the shared weights while another graph needs them).<br>
There is no L=1024 graph in the Release (it would carry its own copy of the weights, 7.5 GB more). Export it yourself if you need it (next section).

To export it yourself (fetch the original checkpoints and convert; needs torch, and is heavy in memory and time):
```bash
uv sync --group export
uv run export/prepare_model.py                       # Qwen3.5-2B-Base + Strands' LoRA and head (about 4.6 GB; pinned revisions, SHA-256 checked)
uv run export/export_onnx.py --seq-len 512           # the first bucket (it owns the weights)
uv run export/export_onnx.py --seq-len 384           # later buckets (on their own they add another 7.5 GB of weights)
uv run export/share_external_data.py models/onnx/torso_L384.onnx   # repoint the weights at torso_L512.onnx.data (byte-identical tensors only)
uv run export/export_vision.py                       # the vision encoder
```
`export_onnx.py` also writes `models/onnx/head.npz` and `graph_meta_L<N>.json`. Copy `tokenizer.json` (from `models/release/`, Strands) and `preprocessor_config.json` (from `models/base/`, Qwen3.5) into `models/onnx/`.

### What the Release holds
`uv run release/make_release.py --dry-run` lists them (no hashing, nothing written). `release/make_release.py` stages `models/` into `dist/release/` (it never uploads).

| Asset | Size |
|---|---:|
| `onnx__torso_L384.onnx` / `onnx__torso_L512.onnx` / `onnx__torso_L1280.onnx` | 167.2 MB / 168.6 MB / 177.1 MB |
| `onnx__torso_L512.onnx.data.part000` to `part002` (the shared weights, 7,527,333,888 bytes, split) | 2,040,109,465 bytes each (1.9 GiB) |
| `onnx__torso_L512.onnx.data.part003` | 1,407,005,493 bytes |
| `onnx__vision_encoder.onnx` / `onnx__vision_encoder.onnx.data` | 5.6 MB / 1,325.7 MB |
| `onnx__head.npz` / `onnx__tokenizer.json` / `onnx__preprocessor_config.json` | 4.2 MB / 20.0 MB / 390 bytes |
| `onnx__graph_meta_L384.json` / `_L512.json` / `_L1280.json` / `onnx__provenance.json` | under 1 KB each |
| `onnx__LICENSE-STRANDS-DECIDER.md` / `onnx__LICENSE-QWEN` | 13 KB / 11 KB |
| `manifest.json` / `SHA256SUMS` | small |

About 9.4 GB in total (18 assets, plus `manifest.json` and `SHA256SUMS`).

# Usage

### Examples (Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
The text demo answers a built-in example (a customer-support ticket: is the customer angry, which team, how urgent). Give your own input like this:
```bash
uv run demo_inference_text.py --state "My card was charged twice." \
    --choice "team=billing,tech" --noul "angry=The customer is angry" --score "urgency=low,medium,high"
```
By default the image demo uses `artifacts/images/sample_chart.png` (a bar chart; it asks which bar is the tallest and whether it is a bar chart). The sample is one picture of the synthetic set of the Wikipedia image benchmark (drawn by a script, `syn-0001`): not a photograph, and nothing from Wikipedia. To use your own pictures, give their paths and the questions (up to 4):
```bash
uv run demo_inference_image.py shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
```

```python
from strands_decide import Decider, choice, noul, score

decider = Decider("models/onnx")          # default: L=384 (change it with seq_len=512 and so on)

decider.decide("I was charged twice for my Pro subscription this month.", {
    "team": choice("Which team?", {"billing": "charges, refunds", "tech": "bugs"}),
    "angry": noul("The customer is angry"),
    "urgency": score("How urgent?", ["low", "medium", "high", "critical"]),
})
```

**One question is one forward pass.** The pointer head reads the hidden state at the `<answer>` position (the query) and at the last token of each option line (the keys) and turns them into probabilities. Nothing is generated.

Pictures ride in the same request (up to 4):
```python
decider.decide("Look at the screenshot.", {...}, images=["shot.png"])
```
`images` accepts file paths, `data:image/...;base64,` URLs, bytes and Pillow images. A picture larger than 448 px on the long side is scaled down to 448 px (196 tokens for a square).

### Is there a person in the camera image?
Grabs frames with OpenCV, asks "is there a person in the picture?" about the newest frame and draws the answer on the live image.
```bash
uv run --group webcam demo_inference_webcam.py                        # camera 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/sample_chart.png   # an image or video file instead of a camera (no person in it, so NO)
```
- The decision runs in its own thread and the live image never stops. The banner on top (green = yes, red = no) and the probability of "yes" update whenever a decision is ready
- Change the question with `--question` (a yes/no question). The picture given to the model is scaled down to 224 px on the long side (`--size`)
- Press `q` or `Esc` to quit. OpenCV is a dependency of this demo only

### Question types
| Type | How | Answer |
|---|---|---|
| `choice` | `choice(instructions, {name: description})` (2 to 255 options) | one option (`choice`, `confidence`, `probabilities`) |
| `score` | `score(instructions, [lowest, ..., highest])` (2 to 10 levels) | expected value on an ordered scale (`score`, `legend`, `confidence`); the first level is the lowest |
| `noul` | `noul("statement")` | whether the statement holds (`noul` = probability of yes); the options are fixed as `false` / `true` |

`state` can be a string or anything JSON-serialisable (a dict, for example).

### Running on a GPU (optional)
The default is the CPU. To run on a GPU (CUDA), install `onnxruntime-gpu` instead of `onnxruntime`. **The two write into the same `onnxruntime/` folder and cannot live together.** With the `gpu` extra, swap them after installing:
```bash
uv sync --extra gpu                                              # also installs onnxruntime-gpu (next to the CPU build)
uv pip install --force-reinstall --no-deps onnxruntime-gpu       # overwrite the folder with the GPU build
uv run --no-sync demo_inference_text.py --device cuda            # run with --no-sync from now on (a plain uv run restores the CPU build)
```
- **Requirements**: put the `bin` folders of CUDA 13 and cuDNN 9 on PATH (`onnxruntime-gpu` does not ship the CUDA DLLs and searches PATH; reopen the terminal afterwards). Windows example (adjust the versions):
  ```text
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
  C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
  ```
  DirectML (Windows; no CUDA needed) works with `onnxruntime-directml` and `--device auto` (not tested in this repository)
- **Choosing**: `--device cpu` (default) / `--device cuda` (CUDA, else CPU) / `--device auto` (CUDA or DirectML when present, else CPU). `--gpu` is the same as `--device auto`. Available in `demo_inference_text.py`, `demo_inference_image.py`, `demo_inference_webcam.py` and `verify/bench.py`
- **No silent fallback to the CPU**: the providers actually used are printed (the last line of the demos, e.g. `providers: CUDAExecutionProvider, CPUExecutionProvider`, is what the session really got; also in the output of `verify/bench.py`). If a GPU was asked for but could not be used (missing DLLs, not enough VRAM, ...), a warning with the reason is printed and the CPU is used. **With `--require-gpu` it stops with an error instead** (with a build that has no GPU provider it stops before loading the model)
```python
from strands_decide import Decider, select_providers

decider = Decider("models/onnx", providers=select_providers("cuda"), require_gpu=True)
print(decider.providers)   # the providers actually used
```
- **VRAM**: the fp32 torso weights alone are about 7.5 GB, plus 1.3 GB for the vision encoder. A 4 GB GPU cannot hold them (check with `--require-gpu` above). The GPU is an optional feature for GPUs with enough VRAM
- Probabilities on a GPU do not match the CPU exactly (operation order). The comparison with the reference is CPU only

**It cannot be said that a GPU is faster.** What I have:
- Measured while this was made (RTX 3050 Ti Laptop, 4 GB VRAM, CUDA 13.4, cuDNN 9.20, onnxruntime-gpu 1.31): a CUDA session can be created (VRAM tops out at about 3.7 GB), but the time per question was on par with the CPU (8 threads): no benefit
- The Wikipedia image benchmark's comparison table has no GPU column for Strands (there is no calibrated measurement)
- Running `--device cuda --require-gpu` in this repository on that environment created a CUDA session (`providers: CUDAExecutionProvider, CPUExecutionProvider`) and the built-in example gave the same output as the CPU. The first call took 165 s, which includes loading the weights: it is not a speed measurement. **An out-of-memory error did not occur on this machine.** The behaviour on OOM (warn and use the CPU / an error with `--require-gpu`) is checked only by unit tests with simulated failures (`tests/test_providers.py`); a real OOM is unverified

# The sequence length is fixed (important)
The model's linear attention walks 64-token chunks in a Python loop, which **the exporter unrolls into a fixed number of iterations**. A graph with a dynamic sequence length cannot be made (a graph exported at one length silently ignores everything past that length).

So the graphs are exported at a fixed length L (a multiple of 64) and the input is **right-padded** to L. Every read position precedes the padding and the model is causal, so the result does not change (`verify/check_padding.py`: max difference 0). **The cost follows L, not the number of real tokens.**

The Release holds L=384, 512 and 1280. **The default is L=384** (`Decider(..., seq_len=384)`; `--seq-len` in the demos and `verify/bench.py`).

- The first request loads the smallest graph that is at least `seq_len`
- When a request does not fit, the runtime switches to **the smallest bucket that does** (the old graph is released first, so two large graphs are never in memory together). It stays on that bucket afterwards
- Only when no bucket fits is a `ValueError` raised, naming the length needed
- The three graphs **share one weights file** (`torso_L384.onnx` and `torso_L1280.onnx` point into `torso_L512.onnx.data`; tensors with identical bytes are read from the same place, so the numbers do not change; checked with `cmp`). **Do not delete `torso_L512.onnx.data`**
- Images lengthen the prompt (one 448 px image is about 140 to 200 tokens; four need about 1024, so the Release's graphs load L=1280)

Peak memory (fp32, 8 threads, 2 text questions; a record from when this was made, with other jobs running, so read it as an upper bound):

| L | Peak RAM |
|---|---:|
| 384 (default) | 7.4 GB |
| 512 | 8.1 GB |
| 1024 (not in the Release) | 12.6 GB |
| 1280 | 15.8 GB |

**Times (seconds) are not listed.** There is no measurement on a quiet, calibrated machine yet (only values taken while other work was running). The very first run, which pages the weights in from disk, is especially slow. To measure, stop other heavy work and use `uv run --no-sync verify/bench.py --seq-len N --questions 2 --runs 5`.

# Verification
```bash
uv run verify/run_all.py                                 # all gates (including those that load the model; about 8 GB of RAM or more)
uv run verify/run_all.py --skip-model                    # skip the gates that load the model (fast)
uv run --group verify verify/run_all.py --full-fuzz      # the tokenizer fuzz at full size, when tokenizers is installed
uv run --group test pytest                               # unit tests (no model needed)
```

| Gate | What it checks | Result | Needs |
|---|---|---|---|
| check_prompt | prompt text, `input_ids` and read positions equal the reference's | 16/16 exact | `models/onnx/tokenizer.json` |
| verify | decisions and probabilities vs the reference (fp32); `--seq-len N` checks each bucket | 16/16 same decision, max abs dp 8.6e-4 | the model |
| check_padding | what sits in the padding cannot move the hidden states at the read positions | max difference 0 | the model |
| verify_image | vision-encoder output, prompt ids with image tokens, read positions and probabilities vs the reference vision engine | 8/8 questions, ids exact, max abs dp 7.3e-4, vision output max abs diff 4.4e-4 | the model + pillow |
| check_vision_inputs | image pixels, bilinear positions and position ids vs transformers | 18/18 (pixels within 1 ulp) | + torch, transformers, `models/base` |
| check_tokenizer_fuzz | the Rust `tokenizers` (ids and offsets) | no id mismatch | + tokenizers |
| check_combining | a combining mark shifts offsets by one against HF, but never an option's read position | no position moves | + tokenizers |

The results are the record from when this implementation was made. `check_prompt`, `check_padding`, `verify`, `verify_image`, `check_tokenizer_fuzz` and `check_combining` were re-run in this repository with the same outcome (`check_vision_inputs` was not re-run: it needs the 4.5 GB base model and torch).

`artifacts/golden.json`, `golden_hidden.npz`, `golden_image.json` and `golden_image.npz` are ground truth made by actually running the reference (the official Strands engine, PyTorch fp32, CPU; for images the vision engine of the official repository's main, its commit recorded in `golden_image.json`). Both run the same weights, so any difference is a bug in the port. To keep every file under 5 MB, `golden_image.npz` stores the vision-encoder output for 3 of the 7 pictures only (the synthetic ones: shapes, screen, chart); the question-level comparison runs on every picture. The photos in `artifacts/images/` are CC0 / public-domain pictures from Wikimedia Commons (`LICENSES.json`).

Known difference: for text with combining marks (for example `a` + U+0301) our character offsets differ from HF's by one (HF reports pre-NFC spans). Token ids are identical and the option read positions do not move (checked by `check_combining`).

### Accuracy (for orientation)
No accuracy was newly measured in this repository. The two figures below are quotations of other measurements.
- The original model card ([`StrandsAgents/strands-decider-2B-hobson-v21`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)) reports accuracy 0.7619 (176/231) on JevBench public (4096-token window); other evaluations are on the card too (values for the PyTorch implementation)
- Accuracy (%) of this ONNX implementation on the Wikipedia image benchmark (`wikipedia-vision-decision-bench`; 8 threads, 224 px):

| Commons en | Commons ja | synthetic en | synthetic ja | reasoning en | reasoning ja |
|---|---|---|---|---|---|
| 98.1 | 94.2 | 92.7 | 92.4 | 50.2 | 48.0 |

These are the 1875 image questions (Commons 1268, synthetic 382, reasoning 225), taken from the benchmark's "before migration" measurement. The reasoning set includes formats with few options where guessing already scores well; read it with chance accuracy in mind.

# Limitations
- The sequence length is fixed. The default is L=384; a larger bucket is loaded when a request does not fit; beyond the largest L (1280 in the Release) an exception is raised (naming the `--seq-len` needed)
- fp32 only (no fp16). No video. At most 4 images per request
- For images, the developers of the original implementation state that the adapter (LoRA) was trained on text only and that image answers come from the base model's (Qwen3.5) vision encoder. The confidence cannot tell a missing image from a present one
- `tokenizer.py` and `protocol.py` are specific to this checkpoint and reject unexpected input with an exception
- Memory hungry (about 8 GB of free memory at least)
- Probabilities on a GPU do not match the CPU exactly; with little VRAM the model does not fit and the CPU is used

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
LICENSE                  # Apache-2.0
pyproject.toml           # dependencies (run time: onnxruntime, numpy, pillow; groups: verify / test / webcam / export)
uv.lock                  # locked dependencies
download_model.py        # fetch the model (GitHub Releases, SHA-256 check, joins split files)
demo_inference_text.py   # text inference demo
demo_inference_image.py  # image inference demo
demo_inference_webcam.py # "is there a person?" on a camera (OpenCV)
strands_decide/          # the inference implementation (light dependencies only; never imports export/)
  runtime.py             #   ORT sessions + right padding + readout + bucket switching + GPU providers
  tokenizer.py           #   pure-Python Qwen byte-level BPE
  protocol.py            #   prompt layout, pointer head, temperatures, confidences
  image.py               #   Pillow decode + smart_resize + patching + vision-encoder inputs + 3-axis M-RoPE
  _regex_translate.py    #   Rust-dialect regex to Python re
  _uniprops.py           #   generated: Unicode range tables
export/                  # heavy dependencies (torch / transformers / onnx)
  prepare_model.py       #   fetch the checkpoints (pinned revisions, SHA-256 checked)
  merge_lora.py          #   fold in the LoRA, read the head
  export_onnx.py         #   the torso (LoRA folded in) + the readout gather, to ONNX
  export_vision.py       #   the vision encoder to ONNX (vision_export_patch.py swaps in the single-image variant)
  share_external_data.py #   share one weights file between buckets (byte-identical tensors only)
  make_golden.py         #   write artifacts/golden.json (runs the questions of cases.py on the reference)
  make_golden_image.py   #   write artifacts/golden_image.* (cases_image.py)
  gen_uniprops.py        #   generate _uniprops.py
  download.py            #   fetch from Hugging Face with the standard library only
release/
  make_release.py        # stage the Release files from models/ (splitting, SHA-256, manifest.json); --dry-run only lists
  notes.md               # the Release's body text
tests/                   # unit tests (no model needed)
verify/                  # verification gates (all at once with run_all.py), bench.py (time and memory)
artifacts/               # golden* (ground truth), images/ (pictures for the demos and tests)
models/                  # ONNX, tokenizer, configs (git-ignored)
```

# Reference
- [StrandsAgents/strands-decider-2B-hobson-v21](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) (the original model: LoRA adapter and pointer head; Apache-2.0)
- [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) (the original inference and training code; Apache-2.0)
- [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (the original model's base; Alibaba Cloud; Apache-2.0)

# License
Strands-Decider-2B-ONNX-CPU is under the [Apache-2.0 license](LICENSE).<br>
The model weights follow the licence of the original model ([StrandsAgents/strands-decider-2B-hobson-v21](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21), Apache-2.0; its base Qwen3.5-2B-Base is Apache-2.0 too). The Release ships both licence texts unchanged (`LICENSE-STRANDS-DECIDER.md`, `LICENSE-QWEN`). Neither repository has a NOTICE file (checked on 2026-10-11).<br>
The code in `strands_decide/` is a numpy port of Strands Decider's inference code (Apache-2.0) and of Hugging Face Transformers' Qwen2-VL image processing and Qwen3.5 vision-position computation (Apache-2.0).

## About the converted files
The ONNX files in the Release are conversions, made with `export/` in this repository, of [Strands Decider 2B](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) (Apache-2.0; a repository that holds only a LoRA adapter and a pointer head), for which no ONNX is published officially. The changes made are:
- The LoRA adapter is folded into the weights of the base model [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (Alibaba Cloud; Apache-2.0) (`W + (alpha / r) * B @ A`) and the text decoder (the torso) is written as an fp32 ONNX graph. The original repository does not contain the base weights
- The graphs are split by fixed sequence length and share one weights file
- The base model's vision encoder (Qwen3.5's, weights unmodified) is written as a single-image ONNX
- The pointer head (a LayerNorm and two projections) is converted to `head.npz` (fp32) and computed in numpy

The weights are exported in fp32 and the decisions were checked to match the original implementation (PyTorch fp32): text 16/16, images 8/8, maximum probability difference 8.6e-4. The original model's training data includes sources licensed Apache-2.0, CC-BY, CC-BY-SA, CC0 and MIT as well as ones tagged `other`, `unknown` or untagged (the list is in the original model card's `LICENSE.md`). Before redistributing, check the original model card and each data licence. The providers of the original model do not guarantee this converted work.
