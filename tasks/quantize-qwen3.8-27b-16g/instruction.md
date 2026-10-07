# Compress Qwen3.8-27B to at most 16 GiB

The fixed BF16 checkpoint is available read-only at `/input/model`.

This task minimizes quality-penalized PPL times decode latency on the GPU allocated for evaluation. PPL must be at most 1.20 times the task-owned BF16 baseline for the same evaluation split; otherwise reward is zero and the submission is excluded from ranking. Capacity, valid inference and independent review remain mandatory.

You have up to 8 hours (28,800 seconds) for the Agent phase, including quantization, public checks and waiting. Final verification has its own separate timeout.

Start from the editable inference template with `cp /app/inference.py /app/submission/inference.py`.

Create a self-contained submission at:

```text
/app/submission/
├── model/
└── inference.py
```

Requirements:

- The final package, including the task-provided official tokenizer, must not exceed 16 GiB. The public Checker reports this final-package size even if your submission omits tokenizer files.
- `model/` must contain compressed text-model weights and configuration loadable by your submitted inference code in the provided environment. You do not need to submit a tokenizer.
- `inference.py` must set `INFERENCE_API_VERSION = 1` and provide `load()` returning an engine with `generate()` and `token_logprobs()`. The Checker executes this submitted implementation, not a separate fixed vLLM model. You may adapt the starter and include helper/kernel source files. See `/opt/quant-docs/runtime/inference-api.md` for the contract. A CLI-only script is insufficient.
- The submission must not contain symbolic links. It must use the supplied packaged model directory and must not depend on external files other than the preinstalled runtime.
- Preserve text-model quality as measured by perplexity on a hidden fixed text corpus. Lower perplexity is better.
- Both lower PPL and lower decode latency improve the objective. Use the same submitted model and causal semantics for both. The 20% relative PPL cap is mandatory; passing it is not an accuracy guarantee.
- For this speed-scored task, the engine must also implement `stream_generate(token_ids, *, max_new_tokens, ignore_eos=False)` for one prompt, yielding exactly one new CPU-ready token per event, before computing the next output token. A submission without this method cannot receive a speed score. Use the same weights and model semantics for generation, PPL and streaming.
- Do not modify `/input/model`. During evaluation, load only submitted weights. Do not read the original checkpoint, hidden evaluation files, score/log files, network services or other models. Do not fabricate probabilities, look ahead to future tokens, skip computation required by the submitted model or use different model semantics for generation and PPL. These rules are subject to independent review; a good measured score is not automatic approval.
- The task owns the tokenizer. For calibration, load it read-only from `/input/model`. Before generation and PPL, the public Checker and hidden Verifier prepare a private package containing the official `tokenizer.json` and `tokenizer_config.json`. Candidate tokenizer files are not used; vocabulary/merge files, special/added-token files, other tokenizer model formats and standalone chat templates are discarded from this package. The Candidate directory stays unchanged, and weight files are reused from its read-only mount without copying their contents. The host wrapper places the same official tokenizer files into the final archived package.
- Compress the supplied model: any compression method is allowed, including quantization, pruning, layer removal, channel reduction and dimension reduction. Do not substitute an unrelated model. Keep the task-owned tokenizer and use the same submitted model semantics for generation, PPL and streaming. Report genuine computation; results remain subject to independent review.
- Evaluation uses text only, but the supplied checkpoint remains complete. Only the text-language components may be optimized. Preserve all supplied visual and other non-text modality modules (including modality encoders, projectors/adapters, and associated configuration) unchanged in the submitted package: do not delete, prune, quantize, replace, or modify their tensor names, shapes, dtypes or values. All retained files count toward the capacity limit; the limits are unchanged. Text-only inference may skip executing these non-text modules. MTP is a text prediction module, not a visual module; its existing retain/quantize/remove permission is unchanged, but speculative decoding remains prohibited. The compressed text model must support genuine generation and token probabilities with the task-owned tokenizer and submission API.

The default `python3` environment contains vLLM 0.28.0 for the starter. You may modify the submitted runtime using installed dependencies or self-contained helpers; modifications installed only in the Agent container do not transfer to the Checker. Store model weights in SafeTensors files and include `model/config.json`; compressed architectures must be loadable by the submitted runtime and compatible with the task-owned tokenizer. LLM Compressor and its compatible dependencies are isolated under `/opt/quantize`; invoke them with `/opt/quantize/bin/python`. You may write and run Python or shell scripts and choose any compression method supported by the installed environment.

## Optional calibration data

A fixed WikiText-2 raw training split is available at
`/opt/calibration/train.parquet`. You may use it to calibrate
data-dependent quantization methods; its use is optional.
Read `/opt/calibration/README.md` for the source and a reproducible sampling example.
This is calibration material, not public Checker text or hidden test data.
Use the tokenizer of the model being quantized. Keep calibration files and
temporary outputs outside `/app/submission`; inference must not depend on them.

## Decode measurement and final reward

The capacity limit applies to the final submission's file size. Inference may use
all available memory on the single GPU allocated to the task.
The `gpu_memory_utilization` argument passed to `load()` is a tunable backend
hint. The Agent may override it in submitted inference code or use a custom
backend's memory management. Actual allocation depends on the backend and
available device resources. Evaluation requests, concurrency and other task
rules remain unchanged.

The checker uses one allocated GPU with the pinned vLLM 0.28.0 runtime. Each request has 2048 input tokens and exactly 128 output tokens,
greedy decoding, concurrency 1. After one warmup request, the same three fixed
prompts run for three rounds. Public and hidden checks use different fixed prompts,
with identical lengths and measurement rules. Hidden prompts are not shipped in
the Agent environment. Each round averages its three decode durations;
the final duration is the median of the three round averages.

The checker timestamps token arrival, not times reported by your code. Decode
duration starts at token 1 and ends at token 128, excluding first-token waiting:
`speed = 127 / decode_duration`. Runtime options are max_model_len=2176,
max_num_seqs=1 and gpu_memory_utilization=0.5 for this stage. The starter enables
compilation/CUDA Graphs; you may optimize the implementation. Disable prefix
reuse and cross-request answer caching; do not buffer a completed answer and
then emit it rapidly. Speculative decoding is prohibited, including MTP, EAGLE,
draft-model and n-gram speculation. Generate output tokens autoregressively,
one at a time; emit each new CPU-ready token before computing the next output
token. Every stream event must contain exactly one token. Do not compute or
verify multiple future output tokens together and then split them into
single-token events. The task-owned parent records each event's receipt time;
stream delivery remains timed. Single-token delivery alone does not prove
compliance with the algorithm restriction; independent implementation review
is still required. All required weights and helpers remain subject to the
package and offline rules.

The measurement protocol is `submitted-decode-single-token-3x3-v3`.

After capacity and valid inference checks pass, minimize:

```text
C = PPL * decode_latency_seconds
m = 1 + 0.5 * clamp((PPL - 1.10 * split_BF16_PPL) / (0.10 * split_BF16_PPL), 0, 1)
C_penalized = C * m
reward = 0 if PPL > 1.20 * split_BF16_PPL else 1 / (1 + C_penalized)
```

The multiplier m is 1 through 10% PPL growth, then increases linearly to 1.5 at 20% (1.25 at 15%).
For submissions passing the PPL cap, reward is a strictly monotone encoding of C_penalized for the harness's 0..1 interface;
it has no Reference anchor or clipped full-score plateau. Exactly 20% passes. Public and hidden use their own measured BF16 baselines, not the quantized Reference. A score
such as 0.1 is not 10% accuracy or 10% of Reference quality. Optimize C_penalized, not a
speed-only score. Public and hidden evaluation use the same objective but
different PPL text and decode prompts. Always inspect the separate PPL and latency.
All raw PPL, decode latency, product, speed and timing variation are retained.
Results also report quality_penalty_multiplier and penalized_ppl_latency_product.
Final `reward.txt` comes from `score.json`, not the diagnostic `ppl.json`.

## Fixed C4 PPL protocol

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
in the Agent image. The optional 9B profile uses separately prepared 9B token IDs.
PPL runtime options: batches of 16 blocks, max_model_len=2049,
max_num_seqs=32, gpu_memory_utilization=0.45.

Protocol, split and scored-token count are saved with each result.

## Public C4 baselines

| Model | C4 PPL (lower is better) |
|---|---:|
| Original 27B BF16 | 9.892151 |
| This task's 16G Reference | 10.413049 |

PPL uses the fixed public C4 256x2048 inputs, one measurement per model.
Decode latency is measured separately on the allocated GPU with three requests per round
and three rounds, taking the median of round means and excluding TTFT.
The product combines the measured PPL and decode latency. Latency results are comparable only within the same evaluation environment.
The quantized Reference provides a diagnostic comparison. Scoring uses the
same-split BF16 PPL to define the penalty start (1.10 times BF16) and quality
limit (1.20 times BF16). BF16 is a measurement baseline; submissions must meet
the capacity limit.


## Optional Qwen3.5-9B debugging resource

When provisioned by the host, `/input/debug-model` contains the read-only
`Qwen/Qwen3.5-9B` BF16 checkpoint and its own official tokenizer. If its
`config.json` is absent, this optional resource is unavailable: skip it and
continue with the formal task. The formal target remains Qwen3.8-27B.

You may first debug your quantization/export/loading/inference pipeline on 9B.
Write its candidate to `/app/debug-submission/model/` and its inference implementation
to `/app/debug-submission/inference.py`, using the same inference API as the formal
task. Use the 9B tokenizer, not the 27B tokenizer, for local calibration.

```bash
python3 /app/check_submission.py /app/debug-submission --debug-9b --package-only
python3 /app/check_submission.py /app/debug-submission --debug-9b
```

This mode evaluates the debug candidate and uses the
same fixed C4 256x2048 PPL and decode protocols,
with separately prepared official 9B token IDs.
It returns `debug_only: true` and separate PPL/latency/product diagnostics, never
a formal reward or quality certificate. Its separate package-size limit is
32 GiB, independent of the formal task's capacity limit.
9B and formal checks share one serialized Checker queue and the same work GPU.
Do not run local GPU jobs or another check while a check is pending.

The original 9B BF16 public debug C4 PPL is **10.736078** (256x2048, one measurement).
The published 9B baseline covers PPL. Use 9B checks to debug the implementation,
and evaluate quality, memory use and speed on the formal 27B model separately.

After debugging, apply the method to the formal checkpoint at `/input/model`;
put ONLY the final 27B candidate in `/app/submission` and use the normal Checker.
The hidden verifier does not mount the 9B resource and evaluates only the formal submission.
Your formal inference must not depend on the debug model or debug submission.

Offline documentation is available at `/opt/quant-docs/`. Start with `README.md`:
Choose methods independently according to the task constraints and measured results.
`runtime/environment.md` covers task paths, `runtime/inference-api.md` defines the submission
interface, `quantization/README.md` covers general principles, method links and export/self-check,
and `runtime/debugging.md` covers compatibility, troubleshooting and source lookup.
See `runtime/custom-encoding.md` for custom encoding/kernel integration and `references/docs/`
for optional official documents. `references/method-index.md` indexes thirteen offline research
papers (searchable text and original PDFs) and their implementation considerations.
Choose references according to your implementation. Use the provided offline
resources and follow the task's download and installation restrictions.

Inspect the input checkpoint's tensor sizes and hypothetical storage savings on CPU:

```bash
python3 /app/model_inventory.py /input/model --json /app/work/model-inventory.json
```

The inventory reads tensor metadata only and does not use the GPU.
Its storage estimates do not predict PPL, runtime memory,
speed, or kernel compatibility.

The public Checker is an optional self-test tool. You may use it or write your
own tests; calling it is not required for submission. It packages the official
tokenizer before running your submitted inference:

```bash
python3 /app/check_submission.py /app/submission
```

Full checks load the model, measure finite PPL and then decode latency,
so they can take several minutes. Each check has a 1200-second evaluator timeout.
All quantization and self-testing must fit within the overall Agent time budget.
Do not wrap this command in a short outer timeout (for example, 30 seconds).
The shared Agent image provides a managed runner. Use one foreground call:

```bash
python3 /app/run_managed_job.py \
  --command 'python3 /app/check_submission.py /app/submission' \
  --log /app/work/managed-public-check.log --timeout 1800
```

The runner waits locally and returns the exit code and log tail when finished.
Do not append `nohup`, `&`, or repeated `sleep`/`ps` queries. The same runner
can supervise quantization commands and bounded GPU-availability waits. Waiting
counts toward the Agent time budget. A runner timeout stops its command and
returns exit 124; the Checker service may still finish a submitted request.
Keep the submission unchanged until that check has finished.

If this runner is unavailable in your Agent environment, use a foreground tool
call with an adequate timeout, or one bounded local wait on a completion file.
Do not repeatedly call the model merely to check whether a process is finished.
Do not launch duplicate checks or modify a submission while its check is pending.

The Checker validates package size, offline generation, and finite PPL, then measures decode latency and applies the linear 1..1.5 cost multiplier over 10%..20% BF16-relative PPL growth, plus the 20% cap. Raw measurements are retained even when the final reward is zero.
The public response includes `public_reward`, `public_ppl`, `decode_latency_seconds`,
`ppl_latency_product`, `quality_penalty_multiplier`, `penalized_ppl_latency_product` and detailed decode measurements. For a CPU-only package and size check:

```bash
python3 /app/check_submission.py /app/submission --package-only
```

There is no Checker call-count limit. The Checker runs in a separate no-network
container with the Candidate mounted read-only.
Its result is diagnostic only; the Verifier uses a separate hidden PPL corpus
and separate hidden decode inputs. Public product measurements do not replace hidden measurements.
Both public and hidden results have `audit_status: pending` and require a separate
review of submitted code and run logs. `valid: true` means automated checks
completed, not that cheating has been ruled out.

`target_met` remains false: there is no automatic full-score or early-stop threshold.
You decide whether to continue optimizing and which Candidate to submit within
the overall time budget. Checker results and results from your own tests are
not accepted as final scores; the Verifier independently evaluates your submission.

## Compression and execution routes

Quantization is not mandatory. You may use pruning, layer/channel/dimension
reduction, knowledge distillation or recovery training, low-rank factorization,
parameter sharing, coded representations, and combinations. Compatible inference
kernel/runtime optimizations are also allowed. Existing methods are valid; novelty
is not required. All methods must preserve the task-owned tokenizer and satisfy
capacity, causal inference, quality, offline and single-token streaming rules.
Speculative decoding remains prohibited. KV/state compression or skipping work
alone does not necessarily shrink the submitted model package.

Start with `/opt/quant-docs/overview.md` for route selection and
`/opt/quant-docs/inference-optimization/README.md` for execution bottlenecks. Dedicated guides cover
pruning, low-rank factorization, parameter sharing, weight encoding, distillation,
conditional computation and training/export
alignment. The method directories under `/opt/quant-docs/` contain offline source PDFs and
searchable text; reference papers do not imply installed implementations or
validated compatibility with this task's model.

Use the supplied calibration/training data for recovery as well as quantization,
and reserve separate texts for checkpoint selection. Never train on Checker or
hidden evaluation inputs. Record training/search budget, the final export recipe,
package size, full-model PPL and real streaming decode timing. A local error or
kernel-speed test alone does not establish full-model quality or speed.

Shared helpers are documented in `/opt/quant-docs/common/README.md`. Browse the direction guides and method index for implementations, and consult `/opt/quant-docs/runtime/tools.md` for their interpreters, offline dependencies and adaptation limits. Choose tools according to the method; no listed implementation is required.

Source availability does not establish model compatibility. Verify exported capacity, PPL and actual streaming speed. Agent tool environments do not transfer into the Checker.


### Optional math diagnostic (no reward)

```bash
python3 /app/check_submission.py /app/submission --quality-only --questions 32
```

This separate optional action checks package validity and runs a fixed public
MATH-500 subset: 32 questions by default, or 64 with `--questions 64`.
It returns `downstream_quality` (correctness, accuracy and truncation), with
`diagnostic_only: true`; it does not run PPL/decode or return a reward.
The default full check and hidden Verifier are unchanged. Use `--debug-9b` and
`/app/debug-submission` together for the debug model. `--quality-only` cannot be
combined with `--package-only`; `--questions` requires `--quality-only`.
The existing evaluator timeout applies; a timeout is not a completed result.
This public mathematics subset does not establish general model capability.
