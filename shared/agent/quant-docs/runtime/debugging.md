# Debugging and compatibility

- [Public self-check workflow](debugging.md#public-checks)
- [Before a full-model run](debugging.md#before-full-run)
- [When results are poor](debugging.md#poor-results)
- [Runtime compatibility](debugging.md#runtime-compatibility)
- [Installed-source lookup](debugging.md#source-index)

<a id="public-checks"></a>

## Public self-check workflow

This section documents the task's public self-check interface.

1. Run the CPU inventory and examine large stored tensors and their dimensions.
2. Save the quantized weights, matching config and inference.py under
   `/app/submission/`. The task supplies the tokenizer; do not submit your own.
3. Optionally run a CPU-only package/size check before a full GPU check:

```bash
python3 /app/check_submission.py /app/submission --package-only
```

4. Optionally run a complete check and inspect its returned JSON:

```bash
python3 /app/check_submission.py /app/submission
```

The Checker is optional and has no call-count limit. You may also write your own
tests. All self-testing counts toward the overall Agent time budget; each Checker
evaluation has a 1200-second timeout. Run one check at a time and keep its submission
unchanged until it finishes. A disconnected client does not cancel the evaluation.
`target_met` is diagnostic only and does not require you to stop optimizing.
Only the independent Verifier determines the final score.
The client uses `/run/quantization-bench/checker.sock`; it needs no network URL.

| Symptom | Inspect first |
| --- | --- |
| Size gate fails | Actual files under submission, including duplicates and intermediate outputs |
| Missing tensor or scale | Weight names, shard index and saved quantization config |
| Unsupported shape/scheme | Runtime traceback and selected quantization implementation |
| Works locally but fails in Checker | Interpreter versions, unsubmitted package changes and outside-file dependencies |
| Bad generation or high public PPL | Saved weights/config correspondence and correct tokenizer assets |

Pruning and other structural compression are allowed. Keep the saved configuration,
weights and submitted runtime consistent with one another and compatible with
the task-owned tokenizer. Use the same model semantics for generation, PPL and streaming.
Fix the first relevant exception in the traceback; the final engine-start failure
often only reports that an earlier operation failed.

<a id="before-full-run"></a>

## Before a full-model run

Check the installed loader's format, packing and fused-layer requirements before
committing to a recipe. Where useful, test quantization, export, reload and inference
on representative layers or a smaller model using the intended runtime.
Use the formal Checker on a complete submission to verify full-model
compatibility, quality, capacity and speed.

Budget time for quantization, saving the final model and checking it.
Use the task-provided BF16 baseline for the same model, evaluation split and
protocol. Calibration is provided at
`/opt/calibration/train.parquet`; its README documents sampling. This training
data is separate from public/hidden C4 evaluation inputs. Data preparation and
calibration count toward the solving budget.

<a id="poor-results"></a>

## When it runs but the result is poor

- **High PPL:** first check that config/weights came from the same export,
  calibration used the right tokenizer, and the intended layers/schemes were
  selected. Then change one thing at a time: calibration coverage, group size,
  bit width or higher-precision exclusions. Use controlled comparisons to
  identify the cause.
- **Small but slow:** inspect the selected inference kernel, fallbacks, fused-layer
  constraints and precision conversions. Use Checker decode latency; export time,
  model-load time and ad-hoc wall time measure different things.
- **Quantization OOM:** try sequential calibration/offloading and a smaller
  calibration batch first. Sample count, sequence length and Hessian storage
  affect resources differently. Estimate GPTQ memory separately from inference.
- **Local success, Checker failure:** end quantization and reload the saved
  checkpoint with all required helper files. Inspect the first concrete traceback.
  Run both package checks and inference checks.

See the [9B workflow](../quantization/README.md#reload) for provisioned data and actual Checker
commands.

<a id="runtime-compatibility"></a>

## Runtime compatibility

See [source lookup](debugging.md#source-index) for installed loader/backend entry points
and [upstream/vllm-fp8.md](../references/docs/vllm-fp8.md) for the versioned offline FP8 guide.

Optional: [vLLM v0.28.0 CustomOp — offline upstream reference](../references/docs/vllm-custom-op.md)
for registering custom operations and integrating your own kernels. Read its
task note about worker processes, CUDA dispatch and self-contained submissions.

Target runtime: vLLM 0.28.0 on the allocated GPU. Check format compatibility
for the actual layers, shapes and architecture.

| Choice | What must agree with the loader |
| --- | --- |
| FP8 weights and activations | Quantization config, scales, layer dimensions and selected kernel |
| Integer weight quantization | Bit width, packing format, group size and scale/zero-point layout |
| Mixed precision | Configuration and actual tensor dtypes for each included/excluded layer |
| Fused projections | Compatible schemes for the component projections the runtime merges |

LLM Compressor saves compressed-tensors checkpoints for deployment in vLLM.
Different schemes still need their corresponding implementations; a `.safetensors`
extension alone says nothing about which quantization kernel is required.

The vLLM 0.28.0 FP8 guide describes the runtime's FP8 support and a data-free scheme
with per-channel weight scales and per-token dynamic activation scales. It also
recommends separate Python environments for quantization and vLLM evaluation.
Use the task's preinstalled interpreters and local resources, following its
offline restrictions.

To inspect the exact loader available here without changing it:

```bash
python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm.model_executor.layers.quantization.compressed_tensors").origin)'
```

The public Checker runs your submitted inference implementation in a separate
container with fixed preinstalled dependencies. A successful import in the
quantization interpreter does not establish runtime compatibility. The task
evaluates saved Candidate files; modifications to installed packages in the
solving container do not become part of those files.

### Integer backend selection

Do not infer all INT3 support from Marlin or TritonW4A16 alone. This runtime also
contains `HummingLinearKernel`; supported compressed-tensors integer checkpoints
can use it through the existing vLLM loader, without writing a replacement kernel.
Check the installed `compressed_tensors_wNa16.py`, `choose_mp_linear_kernel`,
and `mixed_precision/humming.py` selection and compatibility checks for the
actual packing layout, group size and shape.
Confirm the selected backend in the loading log and test the saved checkpoint
with the public Checker before concluding that custom kernels are necessary.

For the scoring formula, PPL cap and latency protocol, follow the task
instruction. Public and hidden checks use separate inputs and anchors. A small
checkpoint or a good PPL alone does not prove faster inference or lower peak GPU
memory.

<a id="source-index"></a>

## Installed-source lookup

Use source from the installed version when a summary is insufficient. These
entry points match LLM Compressor 0.13.0 and vLLM 0.28.0; transitive dependency
versions may differ, so inspect them instead of assuming their packing layouts.
No model weights need to be loaded just to locate a Python module.

| Question | Interpreter | Module/file or symbol to inspect |
| --- | --- | --- |
| How does data-free quantization export? | `/opt/quantize/bin/python` | `llmcompressor.model_free_ptq`: signature and source file |
| How do calibration and saving work? | `/opt/quantize/bin/python` | `llmcompressor.oneshot`, `GPTQModifier` |
| What are the scheme fields? | `/opt/quantize/bin/python` | `compressed_tensors.quantization.QuantizationArgs`, `QuantizationScheme` |
| How are low-bit values packed? | Both, separately | `compressed_tensors.compressors.pack_quantized.helpers` |
| How is a compressed-tensors scheme selected? | `python3` | `vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors` |
| What integer tensor shapes does the loader expect? | `python3` | `...compressed_tensors.schemes.compressed_tensors_wNa16`, `create_weights` |
| Which linear kernel is selected? | `python3` | `vllm.model_executor.kernels.linear`, `choose_mp_linear_kernel` |
| What can a backend implement? | `python3` | `vllm.model_executor.kernels.linear.mixed_precision.humming`, corresponding Marlin/Machete classes and `can_implement` |
| Which projections are merged? | `python3` | Actual model class under `vllm.model_executor.models`; `packed_modules_mapping`, `load_weights`; then the quantization loader's fused-target checks |
| How do custom operations dispatch? | `python3` | `vllm.model_executor.custom_op`; [CustomOp reference](../references/docs/vllm-custom-op.md) |

Expand the ellipsis above to `vllm.model_executor.layers.quantization` for imports.
Locate the architecture implementation using the local `config.json` and the
runtime's model registry. Follow imported helper symbols when the implementation
delegates to another module.

### Locate functions and inspect signatures

```bash
/opt/quantize/bin/python - <<'PY'
import inspect
import importlib.metadata as metadata
from llmcompressor import model_free_ptq, oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
for name in ('llmcompressor', 'compressed-tensors', 'transformers'):
    print(name, metadata.version(name))
for obj in (model_free_ptq, oneshot, GPTQModifier):
    print(obj.__name__, inspect.signature(obj), inspect.getsourcefile(obj))
PY
```

Find the installed loader directory, then use `rg`/`grep` and `sed` on its source:

```bash
python3 - <<'PY'
import importlib.util
for name in (
    'vllm.model_executor.layers.quantization.compressed_tensors',
    'vllm.model_executor.kernels.linear',
    'vllm.model_executor.models',
):
    spec = importlib.util.find_spec(name)
    print(name, spec.origin if spec else 'not installed')
PY
```

These commands locate modules and may initialize library dependencies.
Inspect the available backends, then confirm the selected kernel through loading
logs and inference on the exported checkpoint. Include required custom code
in the submission so Checker can load it.

The versioned [upstream documents](../references/docs/README.md) and installed source are
technical references; the task instruction remains authoritative for allowed
submissions, evaluation data and scoring. Do not inspect hidden evaluator files.
