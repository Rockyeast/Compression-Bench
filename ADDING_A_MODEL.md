# Adding a new model

This guide is for benchmark maintainers adding a new source model. For another capacity tier of an existing model, follow [the README example](README.md#adding-a-task).

Start from a current task with the closest model, input protocol and scoring policy.
Shared evaluator code, container configuration and identical input arrays can be
reused. A new capacity tier mainly changes task settings; a new model also needs
its own tokenizer, input preparation and measured references.

### 1. Copy a task and register it

Choose a source task with a compatible architecture and evaluation protocol. Copy it to a new directory directly under `tasks/`; use your actual task names:

```bash
bash -e <<'SH'
source_task=tasks/compress-qwen3.8-27b-12g
new_task=tasks/compress-new-model-20g
if [ -e "$new_task" ] || [ -L "$new_task" ]; then
  echo "Task already exists: $new_task" >&2
  exit 1
fi
cp -a "$source_task" "$new_task"
SH
```

`cp -a` preserves the relative Compose and shared-asset symlinks. Keep the new task directly under
`tasks/`, with `shared/` beside it. Update the name in `task.toml`, the model revision
and requirements in `instruction.md`, and the Agent/Verifier timeouts as needed.
Add the task to the Available tasks table after validation. Any copied `solution/` must be adapted and
validated for the new constraints before it serves as a reference.

### 2. Set capacity and scoring

All paths below are relative to the new task. Follow a symlink to edit its shared
source for all linked tasks. For a task-specific change, copy the shared source to a new
named file and update that task's link and Docker `COPY` selector together:

| Setting | Location |
| --- | --- |
| Formal package and resident limits | `evaluator/model_memory_policy.json`: `limit_gib`, read by both Checker and Verifier |
| Optional debug capacity | `environment/public-checker/evaluator-assets/evaluation_profiles.json`: debug profile's `size_limit_gib` |
| Quality and scoring | Public and hidden `evaluator-assets/ppl_config.json`: protocols, `score_policy`, measured baselines and quality limit |
| Decode measurement/reference | Both `evaluator-assets/decode_config.json` files |
| Runtime, packaging and entry-point behavior | `evaluator/runtime_config.json` |

Keep the capacity stated in task instructions aligned with `limit_gib`; package
and resident storage are checked separately against that value. Debug-profile
settings belong to the debug model. Weight encodings are chosen by the submitter;
offline loading, the inference API, capacity and complete resident-storage verification
remain required. `reference_preflight` selects `none`,
`bf16_baseline` or `frozen_reference`. `hidden_reference_fields` controls the extra
reference metadata in hidden PPL reports. Public reporting follows `score_policy`.
Choose an existing scoring policy with its matching `scoring_protocol`; a new protocol
also needs registration in `shared/evaluator/decode_benchmark.py` and boundary tests.
Reuse measured references only when the model revision, tokenizer, input split and
measurement protocol match; new references need public and hidden calibration.

### 3. Select inputs and model assets

For the same model and protocol, keep the existing shared-input `COPY` lines in
`environment/public-checker/Dockerfile` and `tests/Dockerfile`. Each image receives
its selected inputs at the usual `evaluator-assets/` paths.

For a new model, pin the checkpoint and tokenizer revision, update the starter
`environment/inference.py`, calibration material, `source_config.json` and
`reference_models.json` where present. Place its tokenizer under
`shared/evaluator-assets/common/tokenizers/<model>/`, then update both roles'
tokenizer links and `COPY --from=common-assets` selectors. Check
`tokenizer_files` in `evaluator/runtime_config.json` against the new tokenizer's
required files. The private evaluation view links model payloads under `model/`
and copies `.json`, `.py`, `.md` and `.txt` files; official tokenizer assets are
provided separately. This controls copying, not allowed weight formats.
Prepare fixed PPL blocks and decode prompts with separate public and hidden splits;
keep their sampling provenance and model metadata in the task configs.

Store new arrays under `shared/evaluator-assets/public/<input-set>/` and
`shared/evaluator-assets/hidden/<input-set>/`, then select those directories in the
two Dockerfiles. Existing arrays can be reused when token IDs, prefix treatment
and sampling are verified to match. A new PPL prefix/protocol also needs an entry
in `shared/evaluator/ppl_benchmark.py` and a scoring-mask test. Configure optional
debug support consistently across the Agent image, public profiles and public
Dockerfile. Formal and debug profiles share `shared/evaluator/ppl_benchmark.py`;
the selected asset configuration determines the protocol and scored token positions.
Shared Compose already supplies the common and role-specific build contexts.

### 4. Validate and enable

Run `python3 -m unittest discover -s shared/runtime/tests -v` (requires `numpy`, `safetensors` and `torch`). These CPU tests cover submission checks, evaluation stages, scoring and task-tool behavior. Tasks with
`evaluator/runtime_config.json` are discovered automatically; update model-specific
expectations in the tests, including chat smoke and GGUF handling, for new models.

Build the task images using the shared contexts described above. Check that the
Checker receives public inputs and the Verifier receives hidden inputs. On the
target GPU, validate package inspection, model loading, resident storage, complete
public/hidden PPL and streamed decode using a reference candidate. Record whether
the reference satisfies the intended limits before advertising task feasibility.
CPU tests establish wiring and policy behavior; GPU runs establish model support
and measured results. Once validation is complete, use `scripts/run_task.sh` with the new task path,
checkpoint path and a fresh run name. The launcher already passes `--force-build`
to Harbor. See [Quick start](README.md#quick-start).

### Updating images after changes

For changes to shared dependencies, task tools or bundled reference material, run:

```bash
bash scripts/build_agent_base.sh --build-arg CUDA_ARCH_LIST="<compute-capability>"
```

Replace `<compute-capability>` with one target GPU compute capability in `major.minor` form. The script builds `quantbench-runtime:local` and `quantbench-agent-base:local`; task Dockerfiles use these same local tags. Rebuild them when changing dependencies or target architecture. Pinned upstream versions remain in the Dockerfiles and tool manifest.

Then start a new run with `scripts/run_task.sh`. The launcher already passes
`--force-build` to Harbor to rebuild task images from the updated bases; Docker
can reuse unchanged layers. Changes limited to task settings, inputs or evaluator
code require rebuilding task images only, so the existing base images can be reused.

Place inference dependencies in `shared/runtime/Dockerfile` so Agent, Checker
and Verifier share them. Install compression-only tools in the Agent environments
under `shared/agent/`. During image builds, `verify_runtime.py` checks dependency
versions and GGUF patches. Validate model loading, resident storage, full PPL and
streamed decode on the target GPU when adding support for a new format.

## Completion checklist

- Source checkpoint and tokenizer revisions are fixed and documented.
- Public and hidden images receive their respective input sets.
- Package and resident-storage limits match the instructions.
- Reference calibration records include the inputs, protocol, hardware and measured results.
- Reference loading, full PPL, streamed decode and storage coverage are validated on the target GPU.
- The task is added to the README table with links to its instructions.

The commands above create configuration; they do not establish model compatibility or task feasibility. Calibration depends on the selected backend and protocol and must be recorded separately.

## Configuration and build layout

Each task keeps its capacity policy in `evaluator/`. Shared runtime settings, tokenizer directories and reusable configs appear at the usual task paths as relative symlinks; task-specific configs remain ordinary files. The evaluation pipeline (`evaluate.py`), public entry point (`evaluate_public.py`) and Verifier launcher (`test.sh`) live in `shared/evaluator/`. In `shared/evaluator-assets/`, `common/` holds tokenizers and shared runtime settings; `public/` and `hidden/` keep their inputs and split-specific configs separate. Formal and debug decode settings have independent sources. Docker builds materialize the selected shared sources as ordinary files at each role's existing runtime paths; `.dockerignore` excludes the source links from the task build context. Build Checker and Verifier images with the task root as the context, as configured in their Compose files. The task launcher requests these builds automatically.

Inference, tokenizer packaging, resident storage accounting, PPL measurement and decode scoring are maintained once in `shared/evaluator/`. Each task's `evaluator/runtime_config.json` selects its smoke prompt format, GGUF placeholder handling, tokenizer packaging, reference preflight and hidden reporting fields. Each split's `ppl_config.json` pins the PPL/scoring protocols, thresholds and `score_policy`: `bf16_penalty`, `bf16_cap` or `reference_limit`. The PPL protocol determines which prefix tokens are excluded from scoring; the `selected` list records input sampling provenance. Each task's `evaluator/model_memory_policy.json` supplies the formal package and resident storage limits for both roles. Public debug profiles retain their own limits in `evaluation_profiles.json`. `decode_config.json` holds decode measurement settings and reference timing metadata; result status and gate flags are generated by the evaluator.

### Build contexts and container paths

Compose supplies the shared evaluator, common assets and the role's input directory as additional build contexts. A manual `docker build` needs `--build-context shared-evaluator=/absolute/path/to/Compression-Bench/shared/evaluator`, `--build-context common-assets=/absolute/path/to/Compression-Bench/shared/evaluator-assets/common`, plus `--build-context public-assets=/absolute/path/to/Compression-Bench/shared/evaluator-assets/public` for the Checker, or `--build-context hidden-assets=/absolute/path/to/Compression-Bench/shared/evaluator-assets/hidden` for the Verifier. Task Dockerfiles select the model family; only the Checker receives public math questions and optional debug inputs. Rebuilding task images reuses the existing base images.

Task Compose files are relative symlinks to `shared/compose/agent.yaml` and `shared/compose/verifier.yaml`. Harbor sets the project directory to the selected task’s `environment/` or `tests/`, so `build.context: ..` selects that task’s root. Keep `tasks/` and `shared/` together when copying the repository. For manual Compose commands, pass `--project-directory tasks/<task>/environment` (or `tests`) to preserve this path resolution.

Common task tools live in [`shared/runtime/task_tools/`](shared/runtime/task_tools/): the Checker client, CPU model inventory and submission lock install into `/app/`; the Checker server and optional quality diagnostic install into `/checker/`. Task images inherit these files and supply their own starter, evaluation settings and assets. Agent Dockerfiles retain task-specific debug support and response limits. Tool installers and offline references are maintained in `shared/agent/`.
