# Compress Qwen3.8-27B to at most 12 GiB

Compress the supplied BF16 checkpoint into a complete, offline-loadable model and
inference implementation.

## Objective and scoring

- Complete submission package: **at most 12 GiB**, including weights, configuration, the official tokenizer and required helpers.
- GPU-resident model storage: **at most 12 GiB**, independently of package size.
- Quality limit: **PPL <= 1.20 * split_BF16_PPL**. Exactly 20% growth passes; public and hidden use their own BF16 baselines.

After capacity and valid-inference checks pass, minimize the quality-penalized
PPL times decode latency:

```text
C = PPL * decode_latency_seconds
m = 1 + 0.5 * clamp((PPL - 1.10 * split_BF16_PPL) / (0.10 * split_BF16_PPL), 0, 1)
C_penalized = C * m
reward = 0 if PPL > 1.20 * split_BF16_PPL else 1 / (1 + C_penalized)
```

The multiplier is 1 through 10% PPL growth, 1.25 at 15%, and 1.5 at 20%.
A failed required stage gives zero reward. There is no absolute speed cutoff or
reference-anchored full-score plateau. The quality limit is mandatory; passing it
is not a downstream accuracy guarantee. Raw PPL, latency, product, speed and timing
variation are retained, along with `quality_penalty_multiplier` and
`penalized_ppl_latency_product`.

## Public reference measurements

Public baseline: the supplied Qwen3.8-27B BF16 checkpoint, measured on the public C4 split.

| Model / format | Public C4 PPL ↓ | PPL increase vs BF16 | Public decode (s) ↓ |
| --- | ---: | ---: | ---: |
| Original BF16 | 9.892150681098 | — | — |

The public quality limit is **C4 PPL <= 11.870580817318** (BF16 × 1.20). Hidden evaluation applies the same relative rule to its independently measured BF16 baseline; hidden inputs and metrics remain private during exploration. The evaluator retains full-precision values.

The quality penalty starts at BF16 × 1.10, as defined in the scoring formula above. Reference latency is provided for comparison; decode latency contributes to reward. The BF16 row supplies the quality baseline, and submissions must meet the capacity limits separately. PPL describes language-model quality on this corpus; downstream capabilities require separate evaluation.

Lower PPL and lower decode latency both improve the objective. Reward is not
accuracy or a percentage of reference performance. Public and hidden measurements
use different inputs; latency comparisons require the same evaluation environment.

## Files and environment

- `/input/model/`: original BF16 model weights, pinned to revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and mounted read-only.
- `/app/work/`: scripts, calibration outputs and temporary files.
- `/app/submission/model/`: final model weights and `config.json`.
- `/app/submission/inference.py`: submitted implementation of the inference API.
- `/app/inference.py`: editable starter; copy it into your submission and adapt as needed.
- `/app/model_inventory.py`: CPU-only tensor inventory.
- `/opt/calibration/train.parquet`: optional WikiText-2 raw training text, separate from evaluation data.
- `/opt/quant-docs/README.md`: shared offline research and tool documentation.

Agent, Checker and Verifier share a base inference runtime with vLLM 0.28.0,
Humming and the GGUF loading plugin. Agent-only model-creation tools use separate
Python environments; LLM Compressor is invoked with `/opt/quantize/bin/python`.

Use supplied offline assets and installed dependencies; check tool compatibility
with the target model. Checker and Verifier run without network access. Tool
interpreters and dependencies are listed in `/opt/quant-docs/runtime/tools.md`.
The containers have separate filesystems; package required custom code with the
submission, since Agent-only installations are unavailable in the Checker.
Optional method references start at `/opt/quant-docs/overview.md`.
API and loading guidance is in `/opt/quant-docs/runtime/inference-api.md` and
`/opt/quant-docs/runtime/debugging.md`. Method guides describe available
implementations and model compatibility.

Use the optional calibration text for quantization or recovery training; see
`/opt/calibration/README.md` for its source and sampling example. Use the tokenizer
of the model being compressed. Reserve separate texts for local candidate selection.
Never train on public Checker or hidden evaluation inputs. Keep calibration data
and temporary outputs outside the submission; inference must not depend on them.

Inspect tensor metadata without loading the model onto a GPU:

```bash
python3 /app/model_inventory.py /input/model --json /app/work/model-inventory.json
```

Disk space is shared. Keep only the best submission and at most one temporary
candidate; remove superseded temporary weights before creating another candidate.
Do not copy original BF16 weights into your workspace or delete unrelated files.

## Submission requirements

SafeTensors, GGUF and self-contained custom encodings are allowed. Include `model/config.json` and all required loading/inference helpers in the submission. Your implementation must load the packaged weights offline in the Checker environment and satisfy the submission API, tokenizer, package-size and GPU-resident storage requirements. Format support is not automatically provided by the starter; use dependencies available in the Checker. GPU-resident model storage must remain fully verifiable by the evaluator.

Use the official tokenizer provided by the task; do not modify it. The evaluator supplies it during evaluation, and its size counts toward the final submission package limit.

### Inference API

Implement `INFERENCE_API_VERSION = 1` and
`load(model_dir, *, max_model_len, max_num_seqs, gpu_memory_utilization)` in
`inference.py`. The returned engine must implement `generate`, `token_logprobs`
and `stream_generate`, following `/app/inference.py` and
`/opt/quant-docs/runtime/inference-api.md`. A CLI alone is insufficient.
All three methods must use the same submitted weights, causal semantics and
genuine model probabilities.

`stream_generate(token_ids, *, max_new_tokens, ignore_eos=False)` must emit exactly
one new CPU-ready token per event, before computing the next output token.
A submission without this method cannot receive a speed score.

### Model and method rules

Choose the method independently. Quantization, pruning, layer/channel/dimension
reduction, recovery training, distillation, low-rank factorization, parameter
sharing, custom encodings and combinations are allowed. Compatible inference
and kernel optimizations are allowed. Existing methods are valid; novelty is not
required. All methods must satisfy the capacity, quality and inference rules below.

Build from the supplied BF16 checkpoint at `/input/model`; do not modify it or
substitute another model. Reference checkpoints provide comparison measurements,
not alternative starting weights. Evaluation covers text only. You may remove
visual and other non-text encoders, projectors/adapters and modality-only
configuration. Keep configuration consistent with your loader and retain everything
needed for genuine text generation and token probabilities. Retained non-text
components must remain unchanged and count toward capacity. Image, audio and
video support is not required.

MTP is a text prediction module; it may be retained, quantized or removed, but
speculative decoding remains prohibited.

The package must be self-contained: no symbolic links or external dependencies
other than the preinstalled runtime. During evaluation, load only submitted weights.
Do not access the original checkpoint, other models or network services. Do not
inspect evaluation files, process memory, expected answers, score/log files or timers
to fabricate results. Do not fabricate probabilities, look ahead to future tokens,
replay answers or skip computation required by the submitted model. Automated
measurements remain subject to independent implementation review.

### Package and resident-storage accounting

Both the complete submission package and GPU-resident model storage must be
within the task's stated capacity, independently (1 GiB = 2**30 bytes).
Resident model storage includes packed weights, scales, zero points, codebooks,
weight indices, retained original weights, and all persistent decoded/repacked
or alternative-precision weight copies. Shared backing storage is counted once,
using its full allocation size, including storage outside a visible tensor slice.

KV/state cache pools, temporary activations, CUDA context/Graph overhead, and
non-weight temporary workspaces are outside this model-storage limit.
Tile decoding into temporary kernel workspace is allowed. Relabeling a retained weight copy as workspace/cache, hiding it in
native/global storage, or discarding it only around inspection is prohibited.
The evaluator checks after load and after generation/PPL/decode requests (including
warmup); both load and post-inference evidence are required. The resident representation must remain compliant between inspection points;
full-model expansion requires independent implementation review.

The current evaluator-owned adapter inventories PyTorch storages in a vLLM GPU
worker's model object, including unregistered tensor attributes, and excludes
actual backend KV/state-cache storage by identity. Capacity evidence must come
from this evaluator-owned inventory rather than candidate reports or backend logs. Native
allocations, global/closure-held weight caches, additional model owners, or other
backends require an evaluator-owned adapter and independent review. Acceptance
requires verifiable residency with full inventory coverage. All scores remain
provisional until the existing independent code review confirms inventory coverage.
Report load and post-inference storage evidence in the public and hidden results.
`--package-only` validates files and package size; use a full check to validate resident storage.

`gpu_memory_utilization` controls the backend allocation budget; the resident-storage limit applies separately. Storage inventory RPCs run outside token timing.

## Evaluation protocol

### Runtime parameters

Public and hidden evaluation pass the following runtime hints to the submitted
implementation. PPL batches contain independent input blocks; decode runs one
request at a time.

| Stage | Input batch size | `max_model_len` | `max_num_seqs` | `gpu_memory_utilization` |
|---|---:|---:|---:|---:|
| PPL | 16 blocks | 2049 | 32 | 0.9 |
| Decode | 1 request | 2176 | 1 | 0.9 |

These are the runtime options passed by the Checker. You may adjust backend memory
allocation while keeping resident model storage within the task's capacity limit.

Reference measurements shown above predate the default runtime hint change to
`0.9`; they have not been remeasured under this setting.

### C4 perplexity

This task uses `c4-gptq-256x2048-v1`: 256 independent blocks of
2048 tokens sampled once with seed 0 using GPTQ's `get_c4` document/start
sampling rule. The official task tokenizer produces the frozen token IDs.
There is no overlap or sliding window, and no resampling on each check.
The first token of each block is not scored: 524,288 input tokens and
524,032 scored tokens. PPL is `exp(total negative log likelihood / 524032)`,
not the arithmetic mean of block PPLs. This adopts GPTQ's input protocol,
not its model-execution code: probabilities come from submitted `inference.py`.
Public and hidden checks use disjoint sampled documents from separate C4
validation shards, with identical token budgets. Hidden inputs are not included
in the Agent image.

Protocol, split and scored-token count are saved with each result.

### Single-request decode

Decode uses one allocated GPU, greedy generation, concurrency 1, 2048 input tokens
and exactly 128 output tokens. After one warmup request, three fixed prompts run
for three rounds. The reported latency is the median of the three round mean
durations. Public and hidden checks use different fixed prompts with the same
lengths and measurement rules; hidden prompts are not provided to the Agent.

The evaluator timestamps token arrival, not times reported by submitted code.
Decode duration runs from output token 1 to token 128, excluding time to first
token; throughput is `127 / decode_latency_seconds`. The measurement protocol is
`submitted-decode-single-token-3x3-v3`.

Disable cross-request prefix reuse and answer caching. Do not buffer a completed
answer and then emit it rapidly. **Speculative decoding is prohibited**, including
MTP, EAGLE, draft-model and n-gram speculation. Do not compute or verify multiple
future output tokens together and then split them into single-token events.
Single-token delivery alone does not establish compliance; implementation review
is required. All required weights and helpers remain subject to capacity and
offline-loading rules.

The starter enables compilation/CUDA Graphs.

## Public Checker

The public Checker is optional and has no call-count limit. It runs in a separate
no-network container with the candidate mounted read-only. A full check validates
loading/generation and capacity, measures PPL and decode latency, and applies the
objective above. Inspect quality, latency and storage evidence separately; a local
kernel benchmark alone does not establish full-model quality or speed.

Public checks and hidden verification each have a 30-minute execution timeout.
Release GPU memory held by other experiments before calling the Checker. For a
CPU-only file and package-size check:

```bash
python3 /app/check_submission.py /app/submission --package-only
```

For a full public check, use one foreground managed call:

```bash
python3 /app/run_managed_job.py \
  --command 'python3 /app/check_submission.py /app/submission --timeout 2100' \
  --log /app/work/managed-public-check.log --timeout 2400
```

Use the same managed runner for long compression or recovery commands with a
suitable timeout. It waits locally and returns the exit code and log tail when
finished. Do not append `nohup` or `&`, repeatedly call the model just to poll
logs/GPU status, or launch duplicate checks. If the runner is unavailable, use one
foreground call with an adequate timeout or one bounded local wait.

Keep the candidate unchanged while a check is pending, and do not run competing
GPU jobs. A runner timeout returns exit 124; the Checker service may still finish
the submitted request. Waiting and checking count toward the Agent allowance.
If resources become unavailable, report the concrete failure so the operator can
pause and resume with history preserved.

Public results are diagnostic; the hidden Verifier independently evaluates the
final submission. Both public and hidden results require independent code/log
review (`audit_status: pending`); `valid: true` means automated checks completed,
not final approval. There is no automatic full-score or early-stop threshold
(`target_met` remains false). Record the final export recipe, training/search
budget, package size, PPL and real streaming decode timing with your experiment
results. Hidden `reward.txt` comes from `score.json`, not diagnostic `ppl.json`.

## Optional diagnostics

### Qwen3.5-9B debugging resource

When provisioned by the host, `/input/debug-model` contains the read-only
`Qwen/Qwen3.5-9B` checkpoint and its official tokenizer. If its `config.json`
is absent, skip this optional resource and use the formal target at `/input/model`.
Use the 9B model only to debug compression, export, loading and inference; its
results do not establish quality or speed for the formal target model.

Write the debug candidate to `/app/debug-submission/model/` and its inference
implementation to `/app/debug-submission/inference.py`. Use the 9B tokenizer.

```bash
python3 /app/check_submission.py /app/debug-submission --debug-9b --package-only
python3 /app/check_submission.py /app/debug-submission --debug-9b
```

Debug checks use the separate 9B C4 and decode inputs, a 32 GiB package and
resident-model storage limit, and return diagnostics only, never a formal reward.
They share the formal Checker's GPU, queue and 30-minute execution timeout;
checking and waiting count toward the same eight-hour Agent allowance. Do not
run competing GPU jobs or modify a candidate while its check is pending.
The hidden Verifier evaluates only the formal submission at `/app/submission`.
Formal inference must not depend on the debug model or debug submission.

### Math diagnostic

```bash
python3 /app/check_submission.py /app/submission --quality-only --questions 32
```

This optional check reports correctness, accuracy and output truncation on a
fixed public MATH-500 subset of 32 questions; use `--questions 64` for 64 questions.
Results are diagnostic only: this check does not measure PPL or decode latency,
does not contribute to the formal reward, and does not replace a full evaluation.
It counts toward the eight-hour Agent allowance and has the same 30-minute
execution timeout. Do not combine `--quality-only` with `--package-only`.

## Execution notes
You have eight hours for the Agent phase, cumulative across any operator-managed phases. Hidden verification runs afterward. Choose your approach independently. Preserve only the best submission and at most one temporary candidate. Use installed Python environments for dependencies such as pyarrow; /opt/quantize/bin/python can read the calibration Parquet.
