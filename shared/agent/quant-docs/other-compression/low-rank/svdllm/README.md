# SVD-LLM

The compatibility environment uses Transformers 4.35.2, datasets 2.16.1,
NumPy 1.26.4, PyArrow 20.0.0, sentencepiece 0.2.0, accelerate 0.24.1 and
huggingface-hub 0.36.2. This is a core-compression environment for current Python,
not an exact copy of the author's Python 3.9 environment: obsolete plotting and
optional evaluation/training dependencies are deliberately omitted.

```bash
/opt/svdllm/bin/python /opt/quant-docs/common/author_check.py svdllm
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/opt/quant-docs/other-compression/low-rank/svdllm/upstream \
/opt/svdllm/bin/python /app/work/my_svd_adapter.py
```

In your adapter, import `SVDLLM`. Its `profle_svdllm` collects whitening statistics
from a supported model and local calibration batches; `whitening` performs the
author decomposition/replacement. The smoke uses an identity profile, not real
calibration. The source's `ratio` passed to `whitening` is the **retained factor
parameter fraction**, not the percentage removed. `whitening_local_update` has
additional hard-coded CUDA assumptions; it is not covered by the CPU smoke.

The author's Llama/Mistral/OPT model walkers and old Transformers cannot load
current Qwen hybrid architectures unchanged. Port the algorithm into a modern
model adapter or exchange tensors through SafeTensors across environments.
The smoke's weight roundtrip uses the same already-constructed tiny architecture;
it is not a general compressed-model loader. Do not submit the author's pickle
checkpoint; implement SafeTensors plus architecture/config reconstruction and
the task inference interface.

See [tool environments](../../../runtime/tools.md) and [Tool availability](../../../runtime/tools.md).

## Paper reading notes

Truncation-aware low-rank compression. See [low-rank factorization](../README.md).
[Paper](https://arxiv.org/abs/2403.07378) · [text](paper.txt) · [PDF](paper.pdf).
Count both factors and time the complete replacement path, including extra launches.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../../references/papers.json)

No ready-to-run task adapter is provided; see the implementation/adaptation notes above.

[Tool availability](../../../runtime/tools.md)
