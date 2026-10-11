ONNX files for [Strands-Decider-2B-ONNX-CPU](https://github.com/Kazuhito00/Strands-Decider-2B-ONNX-CPU): Strands Decider 2B (`StrandsAgents/strands-decider-2B-hobson-v21`, Apache-2.0) on onnxruntime + numpy.

Fetch them with `uv run download_model.py` (SHA-256 checked against `manifest.json`; split files are joined). Do not download the assets by hand: the shared weights `onnx__torso_L512.onnx.data` are in 4 parts.

These are converted files (see "About the converted files" in the README): the LoRA adapter is folded into the weights of Qwen/Qwen3.5-2B-Base (Apache-2.0) and the decoder is exported as fp32 ONNX graphs (L=384, 512, 1280; they share one weights file); the Qwen3.5 vision encoder is exported unchanged as a second graph; the pointer head is `head.npz`. Decisions match the PyTorch reference (text 16/16, images 8/8).

Licences: `onnx__LICENSE-STRANDS-DECIDER.md` (Strands Decider model, Apache-2.0) and `onnx__LICENSE-QWEN` (Qwen3.5-2B-Base, Apache-2.0), unchanged.

| Asset | Content |
|---|---|
| `onnx__torso_L{384,512,1280}.onnx` | decoder graphs, one per fixed sequence length |
| `onnx__torso_L512.onnx.data.part000..003` | the shared fp32 weights (7.5 GB), split to stay under 2 GiB |
| `onnx__vision_encoder.onnx`, `.onnx.data` | vision encoder (1.3 GB of weights) |
| `onnx__head.npz`, `onnx__tokenizer.json`, `onnx__preprocessor_config.json`, `onnx__graph_meta_L*.json`, `onnx__provenance.json` | head, tokenizer and settings |
| `manifest.json`, `SHA256SUMS` | file list with sizes and SHA-256, and how the parts are joined |
