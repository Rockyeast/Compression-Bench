# Tool environments

Dependencies are prepared when building the Agent image, before the task starts.
Use the explicit interpreter; environments do not switch automatically. Most share base PyTorch while isolating Transformers and other dependencies.
Unsloth uses its own compatible Torch version; see its limitations below. Never
install their requirements into the base inference Python. Do not fetch packages,
repositories or datasets during the task.

Choose a method based on the task constraints, then consult its guide for
invocation, supported targets and export requirements.

## Available tools and limitations

Entries are listed alphabetically. The availability column distinguishes installed
components, local entry points and source-only references. Check the runtime
limitations when selecting a method. Independent inference servers and application
products are not installed as alternative backends.

| Method | Available environment / entry | Model/runtime limitations |
| --- | --- | --- |
| [ALBERT](../other-compression/parameter-sharing/albert/README.md) | Conceptual reference; no extra environment needed | Sharing a pretrained decoder needs its own implementation and recovery |
| [Alpaca-CoT](../distillation/alpaca-cot/README.md) | Author source plus /opt/compression_misc helper dependencies | Legacy training CLI needs API/model adaptation; optional ChatGLM kernels are not installed. |
| [AutoAWQ](../quantization/autoawq/README.md) | /opt/autoawq | Registered architectures and Transformers 4.51.3; newer models need adapters. |
| [AutoFP8](../quantization/autofp8/README.md) | /opt/autofp8 | Legacy supported-model classes and quantization/export formats need checking. |
| [AutoGGUF](../quantization/autogguf/README.md) | Author source only | Requires a llama.cpp converter/quantizer; that separate backend is not installed. |
| [AutoRound](../quantization/autoround/README.md) | /opt/autoround; local calibration/export entry | Backend groups, model coverage, export/reload/quality |
| [BitBLAS](../quantization/bitblas/README.md) | /opt/bitblas/bin/run-python | Uses isolated CUDA 12 runtime libraries; tensor layouts, packing and target integration remain necessary. |
| [Bitorch Engine](../quantization/bitorch/README.md) | /opt/bitorch; n-bit CUDA extension for the build-selected GPU target | Only qlinear/nbit/cuda is built. Other backends and custom integer-weight training requirements are not supplied. |
| [bitsandbytes](../quantization/bitsandbytes/README.md) | /opt/bitsandbytes | CUDA quantization; optional downloaded CPU kernels not supplied |
| [BMCook](../distillation/bmcook/README.md) | Author source only | BMTrain build hard-codes GPU architectures removed by CUDA 13; its runtime is not installed. |
| [Can my GPU run this LLM?](../common/analysis/gpu-poor/README.md) | Bundled static calculator source/data | The bundled calculator source and data are available for adaptation; the web frontend requires a build step. Measure task memory and performance on the target runtime. |
| [ctxfold](../inference-optimization/dynamic-compute/ctxfold/README.md) | Author source only | Node-based prompt encoding. Fixed evaluation inputs must not be rewritten; reference only for this task. |
| [D2Quant](../quantization/d2quant/README.md) | /opt/d2quant; dual-scale quantizers and mean-shift correction | Dual-scale/DAC components; no ready packed task exporter |
| [EasyLLM](../distillation/easyllm/README.md) | Author source only | Private object-store dependency and old Apex/FlashAttention ABI prevent a complete offline runtime. |
| [Efficient-Tuning-LLMs](../distillation/llamatuner/README.md) | Author source plus /opt/compression_misc helper dependencies | Legacy LLaMA-Tuner CLI and optional DeepSpeed dependencies need adaptation. |
| [GPTAQ](../quantization/gptaq/README.md) | /opt/gptaq; local calibration/export entry | Routed MoE rejected: native/quantized expert tokens are not aligned; dense export/reload/quality still required |
| [GPTQ](../quantization/gptq/README.md) | /opt/gptq; local calibration/export entry | Explicit text targets; upstream tracing, packed export and runtime support |
| [GPTQ-for-LLaMA](../quantization/gptq-for-llama/README.md) | Author source; quantizer usable via /opt/compression_misc | Old model walker, Triton/CUDA kernels and full CLI need porting. |
| [Green-bit-LLM](../quantization/greenbit/README.md) | Author source only | AutoGPTQ dependency fails package preparation with this Python/Torch/CUDA stack. |
| [GreenBit LLaMA](../quantization/greenbit-llama/README.md) | Author source plus /opt/bitorch backend | Old Llama loader and GreenBit checkpoint layout need adaptation; no complete CLI validation. |
| [HQQ](../quantization/hqq/README.md) | /opt/hqq | Core quantization; optional optimized backends need integration |
| [Intel® Neural Compressor](../quantization/neural-compressor/README.md) | /opt/neural_compressor | Torch weight-only APIs available; Intel-specific optional runtimes are not installed. |
| [IntLLaMA](../quantization/intllama/README.md) | Author source only | Author pins Torch 2.0.1/CUDA 11.7 and Transformers 4.29; not compatible with the shared inference stack. |
| [Knowledge Fidelity](../distillation/knowledge-fidelity/README.md) | /opt/knowledge_fidelity | PyPI package available; the linked author repository is unavailable. Integrate training/data/export yourself. |
| [kvpress](../state-compression/kvpress/README.md) | /opt/kvpress | Hugging Face attention/cache integration; GDN recurrent state requires separate handling. |
| [lit-gpt](../distillation/litgpt/README.md) | /opt/litgpt | Local GPT model APIs; checkpoint conversion and target-model adapters remain necessary. |
| [Lit-LLaMA](../distillation/lit-llama/README.md) | /opt/compression_misc; installed author package | Original Llama implementation; target architecture, local data and export need adaptation. |
| [LLaMA-Factory](../distillation/llamafactory/README.md) | /opt/llamafactory; text training dependencies | Audio runtime is not installed. Target-model training/export and optional plugins still need integration. |
| [LLaMA3-Quantization](../common/analysis/llama3-quantization/README.md) | Author quantizer components via /opt/compression_misc | Llama-specific evaluation/loading flow needs adaptation; use task Checker for formal metrics. |
| [llm-awq](../quantization/awq/README.md) | /opt/awq; author INT4 CUDA extension | Legacy model walker and packed export need target-model adaptation. |
| [LLM-Viewer](../common/analysis/llm-viewer/README.md) | Author Python components via /opt/compression_misc | Match architecture and hardware descriptions to the target, then compare roofline estimates with measured speed. |
| [LLMC / LightCompress](../quantization/llmc/README.md) | /opt/llmc plus author source | Quantizer components ready; full author CLI requires omitted evaluation/multimodal plugins |
| [LMCache](../inference-optimization/lmcache/README.md) | /opt/lmcache | Cache APIs available; do not reuse evaluation state across independent requests or alter the task protocol. |
| [Megatron-LM](../distillation/megatron/README.md) | /opt/megatron; megatron-core | Core modules available; complete distributed launcher, Transformer Engine and dataset pipeline are not configured. |
| [MInference](../inference-optimization/dynamic-compute/minference/README.md) | /opt/minference | Import initializes CUDA. Sparse-attention support depends on architecture and attention type. |
| [Minitron](../distillation/minitron/README.md) | Reference plus common pruning/recovery tools | Full author workflow needs NeMo/ModelOpt training and export stack |
| [MixLLM](../quantization/mixllm/README.md) | Pinned source only; runtime not installed | vLLM 0.9.0 patch vs pinned 0.28.0; Python 3.11/Conda build assumptions |
| [ModelOpt](../quantization/modelopt/README.md) | /opt/modelopt | PTQ/QAT APIs; target export and optional training stacks need integration |
| [Optimum Quanto](../quantization/quanto/README.md) | /opt/quanto | PyTorch quantized modules; runtime-compatible export required |
| [PackRat](../inference-optimization/dynamic-compute/packrat/README.md) | Author source only | Node-based prompt encoding. Fixed evaluation inputs must not be rewritten; reference only for this task. |
| [QTIP](../quantization/qtip/README.md) | /opt/qtip; author code and compiled CUDA extension | Llama-specific loader, packed formats and kernel-shape restrictions |
| [ShortGPT](../pruning/shortgpt/README.md) | /opt/quantize; local score_layers.py | Ranking only; actual removal and cache/config reconstruction need adaptation |
| [SigmaScale](../other-compression/low-rank/sigmascale/README.md) | /opt/sigmascale plus author source | Low-rank components; local data and architecture integration required |
| [SINQ](../quantization/sinq/README.md) | /opt/sinq; local quantization entry and author runtime | Target-model export and SINQ runtime integration |
| [SliceGPT](../pruning/slicegpt/README.md) | /opt/slicegpt plus author source | Llama/OPT/Phi adapters; new architectures need adapters |
| [Sparsebit](../quantization/sparsebit/README.md) | Author source only | Author requires Torch below 1.12 and torchvision below 0.13; not installed into modern base Python. |
| [SparseML](../pruning/sparseml/README.md) | Author source only | Published 1.9.0 requires Python below 3.12; needs a separate legacy runtime or port. |
| [SpinQuant](../quantization/spinquant/README.md) | /opt/spinquant; author rotation-training and PTQ entries | Original Llama/training/data flow needs task integration |
| [SVD-LLM](../other-compression/low-rank/svdllm/README.md) | /opt/svdllm; author decomposition code | Port walker for newer architectures; reconstruct saved model |
| [torchao](../quantization/torchao/README.md) | /opt/torchao | Hardware-specific recipes; optional MSLK backend not installed |
| [Unsloth](../distillation/unsloth/README.md) | /opt/unsloth; isolated Torch 2.12 / Transformers 5.5 | GPU import required. Single-GPU text APIs; inherited optional distributed/native packages may have incompatible Torch ABIs. |
| [Wanda / SparseGPT](../pruning/README.md) | /opt/quantize; pruning and export tools | Dense zeros alone do not reduce package bytes or latency |

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

QTIP's extension uses the GPU target selected when building the shared images.
Validate the extension on that target; compilation alone does not establish runtime support.

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
