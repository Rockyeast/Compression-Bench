# Tool environments

Dependencies are prepared when building the Agent image, before the task starts.
Use the explicit interpreter; environments do not switch automatically. Most share base PyTorch while isolating Transformers and other dependencies.
Unsloth uses its own compatible Torch version; see its limitations below. Never
install their requirements into the base inference Python. Do not fetch packages,
repositories or datasets during the task.

## Available tools and limitations

| Method | Available environment / entry | Model/runtime limitations |
| --- | --- | --- |
| AutoRound | /opt/autoround; generic local entry | Backend groups, model coverage, export/reload/quality |
| GPTAQ | /opt/gptaq; registered-model dispatch | Upstream walker support, export/reload/quality |
| SINQ | /opt/sinq; local quantization entry and author runtime | Target-model export and SINQ runtime integration |
| SVD-LLM | /opt/svdllm; author decomposition code | Port walker for newer architectures; reconstruct saved model |
| Wanda / SparseGPT | /opt/quantize; pruning and export tools | Dense zeros alone do not reduce package bytes or latency |
| ShortGPT | /opt/quantize; local score_layers.py | Ranking only; actual removal and cache/config reconstruction need adaptation |
| SpinQuant | /opt/spinquant; author rotation-training and PTQ entries | Original Llama/training/data flow needs task integration |
| D2Quant | /opt/d2quant; dual-scale quantizers and mean-shift correction | Dual-scale/DAC components; no ready packed task exporter |
| QTIP | /opt/qtip; author code and compiled CUDA extension | Llama-specific loader, packed formats and kernel-shape restrictions |
| MixLLM | Pinned source only; runtime not installed | vLLM 0.9.0 patch vs pinned 0.28.0; Python 3.11/Conda build assumptions |
| Minitron | Reference plus common pruning/recovery tools | Full author workflow needs NeMo/ModelOpt training and export stack |
| ALBERT | Conceptual reference; no extra environment needed | Sharing a pretrained decoder needs its own implementation and recovery |
| [ModelOpt](../quantization/modelopt/README.md) | /opt/modelopt | PTQ/QAT APIs; target export and optional training stacks need integration |
| [torchao](../quantization/torchao/README.md) | /opt/torchao | Hardware-specific recipes; optional MSLK backend not installed |
| [HQQ](../quantization/hqq/README.md) | /opt/hqq | Core quantization; optional optimized backends need integration |
| [Optimum Quanto](../quantization/quanto/README.md) | /opt/quanto | PyTorch quantized modules; runtime-compatible export required |
| [bitsandbytes](../quantization/bitsandbytes/README.md) | /opt/bitsandbytes | CUDA quantization; optional downloaded CPU kernels not supplied |
| [LLMC / LightCompress](../quantization/llmc/README.md) | /opt/llmc plus author source | Quantizer components ready; full author CLI requires omitted evaluation/multimodal plugins |
| [SliceGPT](../pruning/slicegpt/README.md) | /opt/slicegpt plus author source | Llama/OPT/Phi adapters; new architectures need adapters |
| [SigmaScale](../other-compression/low-rank/sigmascale/README.md) | /opt/sigmascale plus author source | Low-rank components; local data and architecture integration required |


## Additional tool coverage

The source collection covers the remaining compression, training, cache and analysis
tools from the upstream tool directory. Independent inference servers and application
products are not installed as alternative backends. Source-only rows are available for
reading and adaptation; they are not executable runtime claims.

| Tool | Available entry | Limitations |
| --- | --- | --- |
| [llm-awq](../quantization/awq/README.md) | /opt/awq; author INT4 CUDA extension | Legacy model walker and packed export need target-model adaptation. |
| [GPTQ-for-LLaMA](../quantization/gptq-for-llama/README.md) | Author source; quantizer usable via /opt/compression_misc | Old model walker, Triton/CUDA kernels and full CLI need porting. |
| [AutoAWQ](../quantization/autoawq/README.md) | /opt/autoawq | Registered architectures and Transformers 4.51.3; newer models need adapters. |
| [BitBLAS](../quantization/bitblas/README.md) | /opt/bitblas/bin/run-python | Uses isolated CUDA 12 runtime libraries; tensor layouts, packing and target integration remain necessary. |
| [AutoGGUF](../quantization/autogguf/README.md) | Author source only | Requires a llama.cpp converter/quantizer; that separate backend is not installed. |
| [AutoFP8](../quantization/autofp8/README.md) | /opt/autofp8 | Legacy supported-model classes and quantization/export formats need checking. |
| [Bitorch Engine](../quantization/bitorch/README.md) | /opt/bitorch; n-bit CUDA extension for SM90 | Only qlinear/nbit/cuda is built. Other backends and custom integer-weight training requirements are not supplied. |
| [Green-bit-LLM](../quantization/greenbit/README.md) | Author source only | AutoGPTQ dependency fails package preparation with this Python/Torch/CUDA stack. |
| [GreenBit LLaMA](../quantization/greenbit-llama/README.md) | Author source plus /opt/bitorch backend | Old Llama loader and GreenBit checkpoint layout need adaptation; no complete CLI validation. |
| [IntLLaMA](../quantization/intllama/README.md) | Author source only | Author pins Torch 2.0.1/CUDA 11.7 and Transformers 4.29; not compatible with the shared inference stack. |
| [Intel® Neural Compressor](../quantization/neural-compressor/README.md) | /opt/neural_compressor | Torch weight-only APIs available; Intel-specific optional runtimes are not installed. |
| [SparseML](../pruning/sparseml/README.md) | Author source only | Published 1.9.0 requires Python below 3.12; needs a separate legacy runtime or port. |
| [Sparsebit](../quantization/sparsebit/README.md) | Author source only | Author requires Torch below 1.12 and torchvision below 0.13; not installed into modern base Python. |
| [BMCook](../distillation/bmcook/README.md) | Author source only | BMTrain build hard-codes GPU architectures removed by CUDA 13; its runtime is not installed. |
| [Knowledge Fidelity](../distillation/knowledge-fidelity/README.md) | /opt/knowledge_fidelity | PyPI package available; the linked author repository is unavailable. Integrate training/data/export yourself. |
| [LMCache](../inference-optimization/lmcache/README.md) | /opt/lmcache | Cache APIs available; do not reuse evaluation state across independent requests or alter the task protocol. |
| [MInference](../inference-optimization/dynamic-compute/minference/README.md) | /opt/minference | Import initializes CUDA. Sparse-attention support depends on architecture and attention type. |
| [kvpress](../state-compression/kvpress/README.md) | /opt/kvpress | Hugging Face attention/cache integration; GDN recurrent state requires separate handling. |
| [PackRat](../inference-optimization/dynamic-compute/packrat/README.md) | Author source only | Node-based prompt encoding. Fixed evaluation inputs must not be rewritten; reference only for this task. |
| [ctxfold](../inference-optimization/dynamic-compute/ctxfold/README.md) | Author source only | Node-based prompt encoding. Fixed evaluation inputs must not be rewritten; reference only for this task. |
| [Unsloth](../distillation/unsloth/README.md) | /opt/unsloth; isolated Torch 2.12 / Transformers 5.5 | GPU import required. Single-GPU text APIs; inherited optional distributed/native packages may have incompatible Torch ABIs. |
| [LLaMA-Factory](../distillation/llamafactory/README.md) | /opt/llamafactory; text training dependencies | Audio runtime is not installed. Target-model training/export and optional plugins still need integration. |
| [Megatron-LM](../distillation/megatron/README.md) | /opt/megatron; megatron-core | Core modules available; complete distributed launcher, Transformer Engine and dataset pipeline are not configured. |
| [lit-gpt](../distillation/litgpt/README.md) | /opt/litgpt | Local GPT model APIs; checkpoint conversion and target-model adapters remain necessary. |
| [Lit-LLaMA](../distillation/lit-llama/README.md) | /opt/compression_misc; installed author package | Original Llama implementation; target architecture, local data and export need adaptation. |
| [Alpaca-CoT](../distillation/alpaca-cot/README.md) | Author source plus /opt/compression_misc helper dependencies | Legacy training CLI needs API/model adaptation; optional ChatGLM kernels are not installed. |
| [Efficient-Tuning-LLMs](../distillation/llamatuner/README.md) | Author source plus /opt/compression_misc helper dependencies | Legacy LLaMA-Tuner CLI and optional DeepSpeed dependencies need adaptation. |
| [EasyLLM](../distillation/easyllm/README.md) | Author source only | Private object-store dependency and old Apex/FlashAttention ABI prevent a complete offline runtime. |
| [Can my GPU run this LLM?](../common/analysis/gpu-poor/README.md) | Bundled static calculator source/data | Web frontend is not built; estimates do not establish measured task memory or performance. |
| [LLM-Viewer](../common/analysis/llm-viewer/README.md) | Author Python components via /opt/compression_misc | Architecture/hardware descriptions must match the target; roofline estimates are not measured speed. |
| [LLaMA3-Quantization](../common/analysis/llama3-quantization/README.md) | Author quantizer components via /opt/compression_misc | Llama-specific evaluation/loading flow needs adaptation; use task Checker for formal metrics. |

See each method's README for original repository links, selected revisions and
license. [Code manifest](../references/code.json) records source repositories, revisions and directories.
ShortGPT source has no license in the inspected snapshot, so only an independently
written scoring tool is provided. Minitron's repository is a model/documentation
index, not an installable compressor.

The old Transformers environments share Torch with the image. Their inherited
vLLM and other inference packages can have unsatisfied dependency metadata;
do not run inference/checker tools through those legacy interpreters. They do not
replace packages in base Python. Optional paper-specific evaluation frameworks
and datasets are omitted: use public training data and the task Checker.

## Verification scope

Build-time imports and numerical checks are offline and use tiny synthetic data.
They do not establish paper reproduction, current-Qwen support, PPL or speed.
GPU extension compilation does not alone establish GPU runtime correctness.

## Running author code

Use `common/run_author.py` to set offline flags before launching the SpinQuant,
D2Quant or QTIP author CLI. See each method's README for arguments and entry paths.
The author's model/data/export assumptions still apply; these launchers do not
automatically adapt the scripts to the task model.

QTIP's provided extension targets SM90. Other GPU architectures require a matching
build; do not assume the shipped binary supports them.

## Explicit invocation

For installed packages, use the named interpreter, for example:

```bash
/opt/neural_compressor/bin/python /app/work/quantize.py
/opt/unsloth/bin/python /app/work/recover.py
/opt/bitblas/bin/run-python /app/work/packed_matmul.py
```

BitBLAS's wrapper sets its private CUDA 12 library search path before starting Python.
It does not change the base CUDA installation. Author scripts remain in each method's
`upstream/` directory; use local models/data and remove automatic network access when
adapting them. `/opt/compression_misc` provides shared Python dependencies for older
training/analysis examples; it does not automatically port their model loaders or CLIs.
Use the base inference interpreter for the task Checker and final inference.
