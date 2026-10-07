# QTIP

Albert Tseng, Qingyao Sun, David Hou and Christopher De Sa. *QTIP: Quantization
with Trellises and Incoherence Processing* (NeurIPS 2024; revised preprint).
[Paper](https://arxiv.org/abs/2406.11235v4) ·
[Implementation](https://github.com/Cornell-RelaxML/qtip).

- **Idea:** jointly encode many weights with trellis-coded quantization, instead
  of independently rounding each weight to a scalar integer grid.
- **Read:** section 3, bitshift trellises, computed/lookup codes and block-wise
  rounding; experiments with and without fine-tuning and the kernel implementation.
- **Possible experiment:** start with a real matrix shape and compare reconstruction
  error, packed size and decoder cost before integrating a model.
- **Implementation work:** calibration, optional fine-tuning, a new packed
  representation and its decoder/kernel. Author code entry points include
  `lib/codebook/bitshift.py`, `lib/linear/quantized_linear.py` and `qtip-kernels/`.
  Follow [custom encoding integration](../../runtime/custom-encoding.md) for the submission
  interface. Trellis bitstreams require their own decoding implementation and
  must satisfy the task's package rules.

## Environment and compatibility

Bundled author code is in [upstream/](upstream/), with its GPL-3.0 license preserved.
The compatibility environment uses `/opt/qtip/bin/python`, Transformers 4.45.2 and
the image's Torch rather than replacing the base inference environment with the
paper's Torch 2.4.0. Hadamard and QTIP extensions are compiled against the image's
Torch/CUDA ABI. The supplied QTIP binary targets SM90; other GPU architectures
require a matching build. Check the actual matrix shapes and code settings before
using a kernel. See [tool limitations](../../runtime/tools.md#available-tools-and-limitations).

The author model implementation is Llama-specific. Packed trellis weights cannot
be loaded as ordinary GPTQ/vLLM weights. Its fast kernels cover particular code
settings and matrix shapes, not arbitrary models. The original fine-tuning scripts
also expect local Hessians/calibration artifacts; those are inputs for the Agent
to generate, not precomputed solutions provided by this benchmark. `lm-eval` and
paper evaluation data are omitted. Use the task Checker for final assessment.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../references/papers.json)

No ready-to-run task adapter is provided; see the implementation/adaptation notes above.

[Tool availability](../../runtime/tools.md)

Offline checks:
`/opt/qtip/bin/python /opt/quant-docs/common/research_check.py qtip`
and on a GPU add `--device cuda`.
Author quantization CLI:
`/opt/qtip/bin/python /opt/quant-docs/common/run_author.py qtip quantize -- --help`.
