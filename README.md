# Compression-Bench

Compression-Bench evaluates **Agent-driven model compression and inference optimization**. Given a fixed model, capacity limit and compute budget, an Agent selects methods, writes code and submits an executable model. Evaluation measures model quality and decode speed.

Agents can use quantization, pruning, structural reduction, low-rank factorization, distillation-based recovery and inference optimization. Shared tools and offline references support method selection and implementation.

## Available tasks

Each task links to its requirements, pinned source model and evaluation protocol.

| Model | Capacity limit (GiB) | Instructions |
| --- | ---: | --- |
| Qwen/Qwen3.8-27B | 12 | [Task](tasks/compress-qwen3.8-27b-12g/instruction.md) |
| | 8 | [Task](tasks/compress-qwen3.8-27b-8g/instruction.md) |
| google/gemma-4-26B-A4B-it | 16 | [Task](tasks/compress-gemma4-26b-a4b-16g/instruction.md) |
| | 10 | [Task](tasks/compress-gemma4-26b-a4b-10g/instruction.md) |
| google/gemma-4-31B-it | 20 | [Task](tasks/compress-gemma4-31b-20g/instruction.md) |
| | 12 | [Task](tasks/compress-gemma4-31b-12g/instruction.md) |
| Qwen/Qwen3.6-35B-A3B | 20 | [Task](tasks/compress-qwen3.6-35b-a3b-20g/instruction.md) |
| | 12 | [Task](tasks/compress-qwen3.6-35b-a3b-12g/instruction.md) |

The listed limit applies independently to the submission package and GPU-resident model storage. Submission size includes weights, inference code and required model assets. Evaluation uses the task-provided official tokenizer, which counts toward the package limit. Resident model storage is limited separately; KV/state caches and temporary workspaces are excluded. Task instructions describe the detailed accounting rules.

## Evaluation

| Component | Measurement |
| --- | --- |
| Capacity | Submission size and the task's resident model storage limit |
| Quality | C4 PPL on 256 × 2048-token inputs, with a task-specific BF16-relative or quantized-reference quality gate |
| Speed | Single-request generation of 128 tokens; time from the first to the last output token, excluding TTFT |
| Repetitions | Warmup followed by three requests per round for three rounds; median of round means |
| Generation | One token per streaming event; speculative decoding, including MTP, is excluded |

The public **Checker** supports Agent self-testing. The hidden **Verifier** uses separate inputs under the same evaluation protocol. Final acceptance also includes code and weight review.

Submissions must pass capacity, quality and inference-validity checks. Rewards combine PPL and decode latency, with higher scores indicating better results. Exact thresholds and scoring formulas are documented in each task.

## Repository layout

| Directory | Contents |
| --- | --- |
| `tasks/` | Task instructions, starter code, task-specific configuration and Dockerfiles |
| `shared/compose/` | Shared Agent/Checker and Verifier container configurations |
| `shared/evaluator/` | Shared submission checks, inference, storage accounting and scoring |
| `shared/evaluator-assets/` | Shared model assets, evaluation inputs and configuration |
| `shared/runtime/` | Shared inference stack and common task tools used by Agent, Checker and Verifier |
| `shared/agent/` | Agent-only creation tools and offline reference library, built on the shared runtime |
| `scripts/` | Image builds, reference preparation and task launching |

Task-specific requirements and configuration live under `tasks/`; common code and resources live under `shared/`. Task images combine these shared components with their selected configuration and inputs.

For configuration and build details, see [Adding a new model](ADDING_A_MODEL.md#configuration-and-build-layout).

Resources: [Adding a task](#adding-a-task) · [Library](shared/agent/quant-docs/README.md) · [Compression overview](shared/agent/quant-docs/overview.md) · [Tool availability](shared/agent/quant-docs/runtime/tools.md).

## Quick start

Prepare a GPU-equipped Linux host, the NVIDIA container runtime, Docker and Harbor with support for this repository's task format. Prepare the model and evaluation assets specified by the task, and configure API authentication for your chosen Harbor Agent.

The build script downloads and verifies missing [reference archives](shared/agent/references.md), then includes the offline library in the Agent image.

Run from the repository root. The example below selects the 12 GiB task; replace the model path, GPU index, Agent and model names:

```bash
# Build shared images on first use or after dependency changes.
bash scripts/build_agent_base.sh --build-arg CUDA_ARCH_LIST="<compute-capability>"

bash scripts/run_task.sh \
  --task tasks/compress-qwen3.8-27b-12g \
  --model-dir /absolute/path/to/pinned-bf16-model \
  --gpu 0 \
  --agent mini-swe-agent \
  --model provider/model-name \
  --run-name example-12g
```

The launcher creates `jobs/<run-name>/`, its work and submission directories, and the submission lock; it sets the host user/group IDs and starts one Harbor attempt. Use a new run name each time. Relative paths are resolved from your current directory. `--model` selects the Agent's API model; `--model-dir` selects the BF16 checkpoint to compress. API authentication is configured separately for the chosen Agent. Shared images must already be built; the launcher builds the task images. Failed runs are retained for inspection.

Select another task from the table above. All tasks share the inference and Agent-tool base images. Build them on first use or after shared dependency changes, specifying the target GPU compute capability as `major.minor` in place of `<compute-capability>`. The selected target is used for GGUF and Agent CUDA extensions; the build host need not expose a GPU. Changing this target requires rebuilding the bases and task images; dependency and kernel support must be verified on the target GPU.

Supply the checkpoint pinned by the selected task. The launcher mounts the source model read-only, prepares `/app/submission` for the Agent's weights and inference code, and supplies the Compose environment variables.

Agents can call `python3 /app/check_submission.py /app/submission` directly, or use `/app/run_managed_job.py` to manage long-command waiting, logs and timeouts.

<details>
<summary>Optional 9B debug model</summary>

For Qwen-series tasks only (Qwen3.6 and Qwen3.8), Qwen3.5-9B can be provided for testing compression, export and inference workflows. Gemma tasks do not support `--debug-9b`; use their supplied target model for debugging. Append this option to the `scripts/run_task.sh` command above:

```bash
--debug-model-dir /absolute/path/to/qwen3.5-9b-bf16
```

The launcher creates the debug submission directory and passes the optional Compose configuration to Harbor.

This mounts the debug model and submission directory for the Agent and public Checker. Agents use `--debug-9b` for debug-model checks; the hidden Verifier evaluates the main submission.

</details>

## Adding a task

For a **new model**, follow [Adding a new model](ADDING_A_MODEL.md). The example
below adds a **20 GiB capacity tier for the existing Qwen3.8-27B model**, using
its 12 GiB task as a starting point. It reuses the original task’s model and evaluation configuration.

### 1. Copy the task and change its capacity

Run from the repository root, with a new destination directory:

```bash
cp -a tasks/compress-qwen3.8-27b-12g \
      tasks/compress-qwen3.8-27b-20g
```

Then edit these three files inside `tasks/compress-qwen3.8-27b-20g/`:

| File | Before | After |
| --- | --- | --- |
| `task.toml` | Task name ending in `12g` | Task name ending in `20g` |
| `evaluator/model_memory_policy.json` | `"limit_gib": 12` | `"limit_gib": 20` |
| `instruction.md` | Package and resident limits of 12 GiB | Both limits set to 20 GiB |

Package and resident storage are checked **separately** against 20 GiB.
The copied quality policy still allows PPL up to 1.20 times each split's BF16
baseline. Raising capacity alone does not tighten the quality threshold.

Keep the shared resource links and Docker selectors unchanged for this example. The optional 9B debug capacity is independent of the
formal task capacity. Historical reference evidence, including its original
capacity tier, remains historical evidence; changing the task limit does not
turn it into a newly validated 20 GiB result.

### 2. Validate before listing the task

Run the CPU regression checks:

```bash
python3 -m unittest discover -s shared/runtime/tests -v
```

These require `numpy`, `safetensors` and `torch` and check configuration
and evaluator behavior. They do not run GPU inference. Inspect the copied task
for stale model or capacity descriptions. Any copied reference solution must be
validated under the new limits.

On the target GPU, evaluate a reference candidate through the public Checker and
hidden Verifier: package size, resident storage, generation, full PPL and streamed
decode must be checked. Keep the validation records, then add the task to the
Available tasks table. A successful file copy or CPU check alone does not establish
that a candidate passes the task.

### 3. Run an Agent experiment

With shared images and API authentication prepared, use the new task path:

```bash
bash scripts/run_task.sh \
  --task tasks/compress-qwen3.8-27b-20g \
  --model-dir /absolute/path/to/pinned-bf16-model \
  --gpu 0 \
  --agent mini-swe-agent \
  --model provider/model-name \
  --run-name example-20g
```

The launcher prepares the run directories and asks Harbor to build the task
images. Existing shared base images can be reused for this capacity-only change.
Changes to shared dependencies or tools require rebuilding the base images.
