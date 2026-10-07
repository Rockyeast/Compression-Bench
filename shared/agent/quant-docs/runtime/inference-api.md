# Submitted inference with independent review

The Agent submits compressed weights plus a runnable inference implementation.
Public/hidden generation, PPL and decode timing execute that implementation.

## Interface

Copy `/app/inference.py` to `/app/submission/inference.py` and keep it or adapt it.
The starter uses vLLM; compatible alternative backends are also supported.

```python
INFERENCE_API_VERSION = 1
def load(model_dir, *, max_model_len, max_num_seqs, gpu_memory_utilization):
    return engine

engine.generate(token_ids, *, max_new_tokens, ignore_eos=False)
engine.token_logprobs(token_ids)
engine.stream_generate(token_ids, *, max_new_tokens, ignore_eos=False)
```

The starter provides `engine.stream_generate(token_ids, *,
max_new_tokens, ignore_eos=False)` for public and hidden decoding-speed scoring.
It takes one prompt and yields a list containing exactly ONE NEW CPU-ready
token ID per event, with no timestamps. Generate output tokens autoregressively,
one at a time; emit each token as soon as it is ready, before computing the next
output token. Speculative decoding is prohibited, including MTP, EAGLE,
draft-model and n-gram speculation. Do not compute or verify multiple future
output tokens together and then split them into single-token events.
`token_logprobs` must report the same submitted model's genuine probabilities.
The Checker and Verifier reject multi-token events at any position. This checks
delivery granularity; independent implementation review is still required to
verify that the candidate does not use speculation or buffered output.
The task-owned parent timestamps receipt of token 1 and token N and computes
`(N - 1) / (last_receipt - first_receipt)`. This excludes time to first token,
and includes stream delivery overhead.
The starter enables compilation/CUDA Graph optimization
(`enforce_eager=False`) and disables prefix caching for repeated-input timing.
The streaming method is required for decoding-speed scoring. The task instruction
defines the measured workload and reward; this document defines only the required
return behavior. Raw timestamps are collected by the task-owned evaluator.

- Inputs use official tokenizer IDs: batches for `generate` and `token_logprobs`,
  one prompt for `stream_generate`. Never retokenize or change IDs.
- `generate` returns only the NEW token IDs as one list per input.
  Use greedy generation, at most max_new_tokens; with ignore_eos=True, return
  exactly that many. Return CPU-ready IDs only after the computation finishes.
- `token_logprobs` returns one aligned list per input: first entry None, then
  natural-log P(x[i] | x[:i]). Values must be finite and nonpositive. No PPL,
  reward or candidate-reported timing fields are accepted.
- All three methods must use the same submitted model, precision and causal semantics.
  Do not fabricate values, inspect future tokens, skip computation for tests,
  use another model, or switch to a special evaluation-only implementation.
- `gpu_memory_utilization` is a tunable backend hint. Submitted code may override
  it or manage memory itself, using all available memory on the task's single
  allocated GPU. The submission capacity limit applies to files. Evaluation
  requests, concurrency and other task rules remain unchanged.
- `load` is called once per engine. A stage may create a fresh engine; calls in
  that stage reuse it. Load only model_dir; do not hardcode submission paths.
- Include required helper/kernel files inside the counted submission. Sibling
  imports work; use the writable working directory for temporary compilation.
  Local package installations do not transfer to the offline Checker.
- Package-format requirements still apply; pruning and other structural compression of the supplied model are allowed. The compressed model must remain compatible with the official tokenizer and inference API, and perform genuine generation, PPL and streaming computation.
  Include a working loader and computation implementation for custom encodings
  inside SafeTensors. See [custom-encoding.md](custom-encoding.md).
- Pure-text inference should not initialize the image processor. The starter
  sets language_model_only=True. INT3/INT4 target groups must be unambiguous.

For a local generation test, run:
```bash
python3 /app/submission/inference.py --model-dir /app/submission/model \
  --tokenizer-dir /input/model --prompt "Hello" --max-new-tokens 16
```

## Ownership and review

The task controls tokenizer, corpus, requests, PPL aggregation, external timing and reward.
The submitted implementation supplies probabilities. The evaluator checks their
shape and range; independent review verifies the implementation's computation.
Every reported result is `audit_status: pending`,
`review_required: true`, `score_status: provisional`. An independent reviewer
must inspect the exact archived submission, helpers, trajectory and run logs.
Treat `valid: true` and the numeric Harbor reward as provisional measurements.
Final acceptance requires independent review arranged by the benchmark maintainers.

During evaluation, do not access original weights, hidden corpus/score files,
network services or another model. Do not modify the evaluator or its logs.
Generation, PPL and streamed decoding must be genuine inference using only the submitted artifact.

Evaluation uses offline containers, read-only submission mounts, output checks,
execution timeouts and process cleanup. Results with unresolved review findings
remain provisional.

Timing invokes this same submitted engine after warmup with task-controlled
requests and external timestamps. The exact workload, PPL policy and reward
mapping are defined in the task instruction and evaluator; they are separate
from review approval.
