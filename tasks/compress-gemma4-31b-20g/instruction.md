# Compress google/gemma-4-31B-it to at most 20 GiB

Compress the supplied BF16 checkpoint into a complete, offline-loadable model and
inference implementation.

## Objective and scoring

- Complete submission package: **at most 20 GiB**, including weights, configuration, the official tokenizer and required helpers.
- GPU-resident model storage: **at most 20 GiB**, independently of package size.
- Public quality limit: **C4 PPL <= 27.382317458472**. Hidden evaluation uses its own measurement of the same quantized reference.
- After capacity, quality and valid-inference checks pass, minimize `PPL * decode_latency_seconds`; reward is `1 / (1 + PPL * decode_latency_seconds)`. A failed required stage gives zero reward. There is no fixed speed cutoff.

## Public reference measurements

Public reference: `cyankiwi/gemma-4-31B-it-AWQ-4bit`, revision `6f1b616c64bca2a05d67fdd6437c5500a8509327`.

| Model / format | Public C4 PPL ↓ | PPL increase vs BF16 | Public decode (s) ↓ |
| --- | ---: | ---: | ---: |
| Original BF16 | 26.177591003164 | — | 2.48592925 |
| AWQ INT4 | 27.382317458472 | +4.6021% | 1.40766700 |

The public quality limit is **C4 PPL <= 27.382317458472**. Hidden evaluation uses independent C4 inputs and its own measurement of the same reference; hidden inputs and metrics remain private during exploration. The evaluator retains full-precision values.

Reference latency is provided for comparison. Acceptance uses the stated capacity, quality and inference-validity gates; decode latency contributes to reward. PPL describes language-model quality on this corpus; downstream capabilities require separate evaluation.

Lower PPL and lower decode latency both improve the objective. Reward is not
accuracy or a percentage of reference performance. Public and hidden measurements
use different inputs; latency comparisons require the same evaluation environment.

## Files and environment

- `/input/model/`: original BF16 model weights, pinned to revision `842da3794eaa0b77d5f08bae87a17459d91ff475` and mounted read-only.
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

PPL uses 256 fixed independent C4 blocks of 2048 tokens. Each block contains the fixed official non-thinking chat prefix (19 tokens) followed by 2029 C4 continuation tokens. Only continuation tokens are scored: 519,424 tokens per split. The fixed user message is `Continue the following text passage.`, rendered with the original chat template and `enable_thinking=False`; its prefix probabilities are excluded. Public and hidden texts come from separate C4 validation shards. PPL is computed by exponentiating the total mean negative log probability across all scored tokens. The Agent receives neither hidden text nor hidden results during exploration.

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

Use this task's supplied model for debugging.

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
