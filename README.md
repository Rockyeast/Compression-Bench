# Compression-Bench

Compression-Bench evaluates **Agent-driven model compression and inference optimization**. Given a fixed model, capacity limit and compute budget, an Agent selects methods, writes code and submits an executable model. Evaluation measures both model quality and decode speed.

Tasks support quantization, pruning, structural reduction, low-rank factorization, distillation-based recovery and inference optimization. Shared tools and offline references are available for Agents to adapt or combine.

## Tasks and scoring

Tasks provide **12, 16, 24 and 32 GiB** submission limits. Each task's `instruction.md` and `task.toml` specify the model, pinned revision and constraints.

| Component | Rule |
| --- | --- |
| Capacity | Final package size, including weights, inference code and the task-provided official tokenizer |
| Quality | C4 256×2048 PPL, compared with the BF16 baseline for the same evaluation split |
| Speed | Single-request generation of 128 tokens; elapsed time between receipt of the first and last token, excluding TTFT |
| Repetitions | After warmup, three fixed requests per round for three rounds; median of round means |
| Generation | Autoregressive computation with one token per streaming event; speculative decoding, including MTP-based speculation, is currently excluded |

The public Checker supports Agent self-testing. The hidden Verifier uses separate evaluation inputs and the same protocol. Final acceptance includes code and weight review.

Higher reward is better:

```text
C = PPL × decode_latency_seconds
m = 1 + 0.5 × clamp((PPL / split_BF16_PPL - 1.10) / 0.10, 0, 1)
reward = 1 / (1 + C × m)
```

The multiplier is 1 for PPL increases of up to 10%, rising linearly to 1.5 between 10% and 20%. PPL increases above 20%, capacity failures or invalid generation receive zero reward. Evaluation reports PPL, decode latency and reward separately.

## Repository layout

| Directory | Contents |
| --- | --- |
| `tasks/` | Task instructions, public Checker and hidden Verifier |
| `shared/agent/` | Shared dependency image, offline reference library and Agent tools |
| `scripts/` | Shared image build and optional debug-model mounts |

References: [Library](shared/agent/quant-docs/README.md) · [Compression overview](shared/agent/quant-docs/overview.md) · [Tool availability](shared/agent/quant-docs/runtime/tools.md).

`shared/agent/setup/tools.json` and `install.py` manage tool installation. Method documentation describes environments, entry points and adaptation limits.

## Quick start

Prepare a GPU-equipped Linux host, the NVIDIA container runtime, Docker and Harbor with support for this repository's task format. Prepare the model and evaluation assets specified by the task, and configure API authentication for your chosen Harbor Agent.

Run from the repository root. The example below selects the 32 GiB task; replace the model path, GPU index, Agent and model names:

```bash
bash scripts/build_agent_base.sh

export BENCH_AGENT=mini-swe-agent
export BENCH_MODEL=provider/model-name
export QUANT_BENCH_MODEL_DIR=/absolute/path/to/pinned-bf16-model
export QUANT_BENCH_GPU_DEVICE=0
export QUANT_BENCH_HOST_UID="$(id -u)"
export QUANT_BENCH_HOST_GID="$(id -g)"
export QUANT_BENCH_RUN_DIR="$PWD/jobs/example-32g"

# Use a new, separate directory for each run.
mkdir -p "$(dirname "$QUANT_BENCH_RUN_DIR")"
mkdir "$QUANT_BENCH_RUN_DIR"
mkdir "$QUANT_BENCH_RUN_DIR/work" "$QUANT_BENCH_RUN_DIR/submission"
mkdir "$QUANT_BENCH_RUN_DIR/work/.agent-home"
touch "$QUANT_BENCH_RUN_DIR/.submission.lock"

harbor run \
  -p tasks/quantize-qwen3.8-27b-32g \
  -a "$BENCH_AGENT" \
  -m "$BENCH_MODEL" \
  --env docker --force-build \
  --n-attempts 1 --n-concurrent 1 --max-retries 0 \
  --job-name harbor --jobs-dir "$QUANT_BENCH_RUN_DIR"
```

Select another capacity by replacing `32g` in the task path with `12g`, `16g` or `24g`. The source model is mounted read-only; the Agent writes weights and inference code to `/app/submission`. Agent and Verifier Docker Compose configurations share the host environment variables above.

Agents can call `python3 /app/check_submission.py /app/submission` directly, or use `/app/run_managed_job.py` to manage long-command waiting, logs and timeouts.

After changing shared dependencies or reference material, rebuild the base image and use `--force-build` to rebuild task images.

### Optional 9B debug model

Qwen3.5-9B can be provided for testing compression, export and inference workflows. Before starting Harbor, run:

```bash
export QUANT_BENCH_DEBUG_MODEL_DIR=/absolute/path/to/qwen3.5-9b-bf16
mkdir -p "$QUANT_BENCH_RUN_DIR/work/debug-submission"
```

Append this option to the `harbor run` command above:

```bash
--extra-docker-compose "$PWD/scripts/debug-model-compose.yaml"
```

This mounts the debug model and submission directory for the Agent and public Checker. Agents use `--debug-9b` for debug-model checks; the hidden Verifier evaluates the main submission.

## Available tasks

| Capacity | Instructions |
| --- | --- |
| 12 GiB | [Task instructions](tasks/quantize-qwen3.8-27b-12g/instruction.md) |
| 16 GiB | [Task instructions](tasks/quantize-qwen3.8-27b-16g/instruction.md) |
| 24 GiB | [Task instructions](tasks/quantize-qwen3.8-27b-24g/instruction.md) |
| 32 GiB | [Task instructions](tasks/quantize-qwen3.8-27b-32g/instruction.md) |
