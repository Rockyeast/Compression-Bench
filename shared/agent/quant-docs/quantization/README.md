# Quantization guide

This is the general guide, not a recommended recipe. Choose the algorithm,
precision and implementation for the task's size, quality and speed constraints.
The method list below is alphabetical, not a ranking. Availability of an example
does not establish that a method is better for the task.

## Schemes and algorithms

A scheme specifies weight/activation precision and how scales are shared.
W8A16 means 8-bit weights and 16-bit activations; W8 alone does not specify
integer versus floating-point storage. Per-channel, per-group and per-block
scales require different metadata and runtime support.

An algorithm chooses the quantization parameters. Methods may use direct
rounding, calibration-based error compensation, learned parameters, rotations,
or mixed precision. The same bit width can produce different quality and speed
with different algorithms, formats and inference kernels.

## Choosing and validating a method

1. Inspect tensor sizes, model structure and the task constraints. Preserve
   modules the task forbids modifying.
2. Check the method's available environment, architecture support, export format
   and inference kernels. Read its specific documentation before running it.
3. If calibration or training is required, use the provided training data and
   the model's tokenizer. Keep evaluation inputs out of calibration and training.
4. Export actual compressed weights and matching configuration. Reload through
   the submission interface; an in-memory or fake-quantized model is not enough.
5. Compare package size, PPL and decode latency with the same evaluation protocol.
   Let measured results determine the choice; no listed algorithm is mandatory.

<a id="inventory"></a>

## Reading the weight inventory

For an originally dense matrix with N elements, ideal b-bit payload is
`ceil(N*b/8)` bytes. The scanner subtracts this from its current storage size.
Add scales, zero points, padding and retained higher-precision weights separately.
INT8 and FP8 have the same raw 8-bit payload size; check quality and runtime
compatibility for each scheme. The 4-bit estimate assumes dense packing.

Prioritize size measurements over tensor counts. A shape divisibility result
only describes the dimensions; the loader and kernel decide actual support.
Quality changes require evaluation. Keep calibration data separate from hidden
evaluation data, which is unavailable during solving.

<a id="parameters"></a>

## Parameters to change deliberately

| Choice | What to check |
| --- | --- |
| Weight and activation precision | Storage, error and exported-format support in the inference kernel. |
| Scale granularity / group size | Quality, metadata cost, divisibility and kernel support. |
| Target modules / exclusions | Task restrictions, retained-weight size and fused-projection compatibility. |
| Calibration samples / sequence length | Data coverage, resource cost and quality after export. |
| Algorithm-specific settings | Consult that method's documentation; parameter names and meanings differ. |

Use `/opt/calibration/train.parquet` where provided; its README describes sampling.
Do not treat a few invented sentences as representative calibration. Keep sample
selection and evaluation protocol fixed when comparing methods.

<a id="reload"></a>

## Export and self-check

Place exported weights and configuration in the submission's `model/`, with a
compatible `inference.py` alongside it. Run the task's public Checker:

```bash
python3 /app/check_submission.py /app/submission --package-only
python3 /app/check_submission.py /app/submission
```

If the optional 9B debug resource is enabled, use a separate debug submission
and add `--debug-9b`. Debug-model results do not establish formal-model quality
or speed. See the [submission interface](../runtime/inference-api.md) and
[troubleshooting guide](../runtime/debugging.md).

## Methods and tools

Each entry describes its own scope and limitations. A listed tool is not
necessarily compatible with every task model or fully validated end to end.

- [autoawq](autoawq/README.md)
- [autofp8](autofp8/README.md)
- [autogguf](autogguf/README.md)
- [autoround](autoround/README.md)
- [awq](awq/README.md)
- [bitblas](bitblas/README.md)
- [bitorch](bitorch/README.md)
- [bitsandbytes](bitsandbytes/README.md)
- [d2quant](d2quant/README.md)
- [gptaq](gptaq/README.md)
- [gptq-for-llama](gptq-for-llama/README.md)
- [greenbit](greenbit/README.md)
- [greenbit-llama](greenbit-llama/README.md)
- [hqq](hqq/README.md)
- [intllama](intllama/README.md)
- [llmc](llmc/README.md)
- [mixllm](mixllm/README.md)
- [modelopt](modelopt/README.md)
- [neural-compressor](neural-compressor/README.md)
- [qtip](qtip/README.md)
- [quanto](quanto/README.md)
- [sinq](sinq/README.md)
- [sparsebit](sparsebit/README.md)
- [spinquant](spinquant/README.md)
- [torchao](torchao/README.md)

See [environment availability](../runtime/tools.md),
[library documentation and API examples](../references/docs/README.md), and the
[offline paper catalogue](papers/README.md).
