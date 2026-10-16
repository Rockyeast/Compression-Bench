# SINQ

This uses the **bundled Git revision** in [source manifest](../../references/code.json), not the PyPI
`sinq` package. Entry scripts add that source directory to the Python path.
Dependencies include Transformers 5.14.1, accelerate 1.14.0, datasets 5.0.1,
SafeTensors 0.8.0, termcolor 3.3.0 and GemLite 0.5.1.post1. Paper-evaluation
packages such as lm-eval are not needed for this entry and are not added.

```bash
/opt/sinq/bin/python /opt/quant-docs/common/author_check.py sinq
# On a GPU, this fails if GemLite is unavailable instead of accepting a fallback:
/opt/sinq/bin/python /opt/quant-docs/common/author_check.py sinq --device cuda
/opt/sinq/bin/python /opt/quant-docs/quantization/sinq/quantize.py \
  --model /input/model --output /app/work/sinq4 --bits 4 --group-size 64
```

Only data-free SINQ is exposed here; A-SINQ's original calibration helper fetches
external data and needs an offline adapter. The CPU smoke tests packed weights,
numerical agreement with dequantized weights and in-memory state restoration.
It does not prove GPU compatibility, model quality, full-model reload or speed.
GemLite queries a GPU during import, so GPU-less Docker builds check its package
version and run SINQ's CPU path. Run the CUDA smoke before relying on acceleration.

Four-bit compatible shapes can use GemLite; other bit widths/shapes may use a
dequantize-plus-matmul fallback. Check `report.json` for the actual module backend.
The author's `AutoSINQHFModel.from_quantized_safetensors` is the corresponding
loader; use a local directory and `local_files_only=True`. SINQ artifacts are not
ordinary GPTQ checkpoints and are not automatically accepted by stock vLLM.
Qwen module coverage, full export/reload, PPL and decode must be checked separately.
Agent-only environments do not transfer into the Checker: include compatible
runtime helpers and use dependencies actually available in the scoring runtime.

See [tool environments](../../runtime/tools.md) and [Tool availability](../../runtime/tools.md).

## Paper reading notes

Lorenz K. Müller, Philippe Bich, Jiawei Zhuang, Ahmet Çelik, Luca Benfenati and
Lukas Cavigelli. *SINQ: Sinkhorn-Normalized Quantization for Calibration-Free
Low-Precision LLM Weights* (ICML 2026; local preprint).
[Paper](https://arxiv.org/abs/2509.22944v4) ·
[Implementation](https://github.com/huawei-csl/SINQ).

- **Idea:** balance row and column magnitudes using an additional scaling axis,
  making low-bit weight quantization less sensitive to outliers without needing
  calibration text in the base method.
- **Read:** the scaling algorithm, quantization/dequantization formulation,
  No-Overhead SINQ and the inference/quantization-time experiments.
- **Possible experiment:** compare uncalibrated quantization with and without
  the scaling step at the same bitwidth and group size.
- **Implementation work:** compute and store/fold the extra scales consistently
  during export and inference. The author's runtime includes Transformers/GemLite;
  its README lists vLLM integration as work in progress. Inspect compatibility
  before assuming that a SINQ artifact is an ordinary GPTQ artifact.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../references/papers.json)

Dedicated entry: [quantize.py](quantize.py).

[Tool availability](../../runtime/tools.md)
