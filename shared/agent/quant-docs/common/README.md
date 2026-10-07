# Offline compression tools

Run the original pruning/recovery/matrix entry points with `/opt/quantize/bin/python`.
For SVD-LLM and SINQ use their separate interpreters in [tool environments](../runtime/tools.md). They reuse the image's
LLM Compressor 0.13.0 and PyTorch; they do not upgrade the inference environment,
install dependencies at runtime, download models or contact an API.

| Entry | Implementation | Actual output / boundary |
| --- | --- | --- |
| `prune.py` | Upstream LLM Compressor Wanda / SparseGPT | Dense tensors containing zeros; **not** a smaller package or automatic sparse speedup |
| `shrink_mlp.py` | Local PyTorch magnitude baseline | Physically smaller gate/up/down matrices and updated uniform MLP width |
| `recover.py` | Local PyTorch logit KD with LoRA updates | Resumable training plus a merged floating-point student; not RL or quantization-aware training |
| `matrix.py` | Local PyTorch truncated SVD baseline | Smaller factors, SafeTensors save/load and `FactorizedLinear`; not full SVD-LLM |
| Method-local `upstream/` directories | Fixed author-source snapshots | Wanda, SparseGPT, SVD-LLM and SINQ implementations to read/adapt; their original model walkers and dependencies are not task-ready adapters |

The baseline transformations and SINQ quantization entry require plain floating-point weights. Packed INT4, FP8 or
custom quantized layers are rejected. Start from `/input/model`, reduce structure
or recover a floating-point student, then optionally use a quantization tool.
If you quantize the exported model, rerun quality checks: recovery before rounding
does not establish quality after rounding. Keep all work products outside submission
until they satisfy capacity and the inference API.

## Portability boundaries

Common means reusable, not architecture-independent in every detail:

- `matrix.py` SVD operates on plain linear matrices without a model-specific walker.
- `prune.py` and `recover.py` load through Transformers AutoModel interfaces;
  callers select actual module names. Library model support still applies.
- `shrink_mlp.py` checks one dense gate/up/down MLP per text layer and uniform
  intermediate_size. This is a structural requirement, not a Qwen-name restriction.
  Fused MLPs, MoE and other layouts need a separate adapter; checks are not bypassed.
- GPTAQ dispatches to registered upstream model definitions. AutoRound uses AutoModel
  loading and configurable exclusions/fusion groups. See each method's README.
- SINQ and SVD-LLM retain the author's supported-model boundaries; bundled source
  has not been rewritten to promise support for arbitrary architectures.

## Quantization entry points

These editable entry points connect installed libraries to local calibration,
export and submission APIs. They are integration examples, not complete paper
reproductions or a guarantee of model quality.

| Entry | Interpreter | Purpose |
| --- | --- | --- |
| [autoround_mixed.py](../quantization/autoround/quantize.py) | `/opt/autoround/bin/python` | Mixed-bit allocation and AutoRound export; see [AutoRound](../quantization/autoround/README.md) |
| [gptaq_quantize.py](../quantization/gptaq/quantize.py) | `/opt/gptaq/bin/python` | GPTAQ or ordinary GPTQ calibration and export; see [GPTAQ](../quantization/gptaq/README.md) |

For LLM Compressor FP8/oneshot recipes, see the
[quantization guide](../quantization/README.md). The
[Tool availability](../runtime/tools.md) distinguishes bundled tools from methods requiring
additional implementation. These scripts do not install packages or download
models at runtime. A successful export still needs reload, PPL, generation and
streaming-speed checks with the task's Checker.

Optional math diagnostics use the same Checker entry point:

```bash
python3 /app/check_submission.py /app/submission --quality-only --questions 32
```

Use `--questions 64` for the larger fixed public MATH-500 subset. This action
checks the package, then reports correctness and truncation through the isolated
Checker with its official tokenizer. It does not run PPL/decode scoring or change
reward. The default Checker action is unchanged. A timeout/failure is not a
completed quality result. Questions and answers are public diagnostics, not hidden
score estimates. Budget this optional run within the task's evaluator timeout.

## Calibration and selection data

Tools take JSON token blocks, e.g. `[[12, 87, 9, 6], [51, 8, 6, 7]]`.
Generate real blocks from the supplied public training split with the fixed task
tokenizer. See the task calibration sampling example at `/opt/calibration/README.md`.
Partition source texts into disjoint training and validation splits **before**
tokenization; do not reuse Checker or hidden evaluation inputs. `recover.py`
rejects identical blocks across splits but cannot detect all overlapping source
spans. Use a separate independent final evaluation after checkpoint selection.

The following examples assume `/app/work/train.json` and
`/app/work/validation.json` have already been created. Choose fresh output paths.

## Physical MLP reduction

```bash
/opt/quantize/bin/python /opt/quant-docs/common/shrink_mlp.py \
  --model /input/model --output /app/work/pruned --width 12288
```

Choose a width smaller than the actual `text_config.intermediate_size`; 12288 is
an example, not a prescribed optimum. This baseline ranks channels by the product
of gate/up row norms and down column norms, keeps identical channel indices in all
three matrices, and changes every text layer to the same width. It rejects MoE,
unknown layouts and mismatched dimensions. It leaves attention/GDN unchanged.
For per-layer widths or head/depth pruning, write a matching config/loader rather
than assuming the stock architecture can represent arbitrary shapes.

## Wanda and SparseGPT

```bash
/opt/quantize/bin/python /opt/quant-docs/common/prune.py \
  --model /input/model --calibration /app/work/train.json \
  --output /app/work/wanda --method wanda --sparsity 0.5 \
  --targets 're:.*mlp\.(gate_proj|up_proj|down_proj)$'
```

Use `--method sparsegpt` for Hessian-based compensation. `--mask 2:4` requires
`--sparsity 0.5` and compatible dimensions. The basic calibration pipeline can
hold Hessians for many targets concurrently; on large models, start with a few
explicit module targets and measure memory, or adapt the upstream sequential
pipeline to the actual architecture. **Zeroing a dense weight tensor saves no
model bytes.** Combine with supported packing/quantization or build a valid sparse
representation/runtime; do not submit this alone expecting a smaller model.

## Distillation and recovery

```bash
/opt/quantize/bin/python /opt/quant-docs/common/recover.py \
  --teacher /input/model --student /app/work/pruned/model \
  --train /app/work/train.json --validation /app/work/validation.json \
  --output /app/work/recovered --rank 16 --steps 64 --eval-every 8 \
  --max-minutes 20 \
  --targets '.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)'
```

Inspect `model.named_modules()` first: the full regex selects actual linear module
names, and the tool logs the selected list and trainable parameter count. It is
not restricted to MLP. It does not alter convolution, recurrent states, norms or
embeddings. Teacher and student must use the same task vocabulary and tokenizer.

The loss scores **every valid next-token position**, using teacher-to-student KL
(optional `--ce-weight`) and a frozen teacher. Model eval mode disables dropout
without disabling gradients. The tool selects the best branch on held-out NLL,
including the unchanged step-zero student, then reports branch and merged NLL
separately. `result.json` is a training diagnostic, not formal task PPL.

Both models are resident; LoRA reduces gradient/optimizer storage but not the
base-model memory. Start with short sequences and a memory preflight. You may use
`--teacher-device cpu` if necessary, at substantial throughput cost. The loop time
budget is checked after steps; model loading, validation and export take extra
time. Full-vocabulary logits also consume memory.

Resume into a fresh output directory using `--resume /app/work/recovered/resume.pt`
and increase `--steps`. Keep original teacher/student paths, model contents,
training/validation blocks and recipe unchanged. The checkpoint includes current
LoRA parameters, optimizer, data position, RNG and selected best parameters;
`--steps` is the cumulative target. Paths/data/settings are checked; model weight
files must not be overwritten between runs. The tool logs actual completed steps.

`model/` is a merged floating-point checkpoint, not an adapter that loads the
teacher. Reload it in a fresh process. BF16 merging can change rounding; compare
merged quality before any later FP8/INT4 export. Do not ship `resume.pt`, teacher
weights or training data in the candidate package.

## Low-rank matrices

```bash
/opt/quantize/bin/python /opt/quant-docs/common/matrix.py \
  --weights /app/work/weights.safetensors --key layer.weight \
  --rank 256 --output /app/work/factors
```

This exports one matrix as two factors with fewer parameters; it does not create
a complete language model. Copy `matrix.py` **and `common.py`** into the submission
if importing `FactorizedLinear` there. Replace the corresponding runtime module,
save/load the remaining weights, and account for factor metadata. Stock vLLM does
not automatically understand this factor format. Two smaller matmuls are not
guaranteed faster than one dense matmul. Read SVD-LLM's source for data-aware
whitening and reconstruction; plain truncated SVD is only the control baseline.

## Author sources and compatibility

`references/code.json` records each repository commit and source directory.
Original source/docs/licenses are preserved where supplied by the author. Older
selections omit image assets; additional tool snapshots may retain them. Model
checkpoints and installed environments are not bundled in source directories. No
upstream script is executed automatically. Some original scripts download data or
assume Llama/OPT and older Transformers: adapt those parts for offline Qwen use.

- [Wanda](https://github.com/locuslab/wanda): `/opt/quant-docs/pruning/wanda/upstream/` (MIT).
- [SparseGPT](https://github.com/IST-DASLab/sparsegpt): `/opt/quant-docs/pruning/sparsegpt/upstream/` (Apache-2.0).
- [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM): `/opt/quant-docs/other-compression/low-rank/svdllm/upstream/` (Apache-2.0).
- [Minitron](https://github.com/NVlabs/Minitron): linked reference only; no repository
  license was found in the inspected snapshot, so its code is not redistributed.

Passing small-model tests proves tooling behavior only. Model-specific 27B load,
full PPL, size, prefill/decode consistency and streaming performance remain
candidate checks. New task calibration/readiness requirements are unchanged.

Additional isolated author tools and ShortGPT offline notes: [environment guide](../runtime/tools.md).

`extra_tool_check.py TOOL --device cuda` exercises a tiny offline component using
that tool's `/opt/TOOL/bin/python`. See [environments](../runtime/tools.md).
It is not an end-to-end model test.

Additional analysis source: [GPU memory calculator](analysis/gpu-poor/README.md), [LLM-Viewer](analysis/llm-viewer/README.md), [LLaMA3 quantization](analysis/llama3-quantization/README.md).
