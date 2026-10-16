#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
usage() {
  cat <<'EOF'
Usage: bash scripts/run_task.sh --task TASK --model-dir DIR --gpu DEVICE \
  --agent AGENT --model PROVIDER/MODEL --run-name NAME [--debug-model-dir DIR]

Creates a new jobs/NAME directory and starts one Harbor Docker attempt.
Relative paths are resolved from the caller's working directory.
Build the shared images and configure model API authentication beforehand.
EOF
}
fail() { echo "Error: $*" >&2; exit 1; }
task= model_dir= gpu= agent= model= run_name= debug_dir=
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --task|--model-dir|--gpu|--agent|--model|--run-name|--debug-model-dir)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
      case "$1" in
        --task) task=$2 ;; --model-dir) model_dir=$2 ;; --gpu) gpu=$2 ;;
        --agent) agent=$2 ;; --model) model=$2 ;; --run-name) run_name=$2 ;;
        --debug-model-dir) debug_dir=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ -n "$task" && -n "$model_dir" && -n "$gpu" && -n "$agent" && -n "$model" && -n "$run_name" ]] || { usage >&2; fail 'All required options must be supplied'; }
[[ "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || fail 'Run name must start with a letter or digit and contain only letters, digits, dots, underscores or hyphens'
[[ -f "$task/task.toml" && -f "$task/environment/docker-compose.yaml" ]] || fail "Invalid task directory: $task"
[[ -d "$model_dir" ]] || fail "Model directory does not exist: $model_dir"
command -v harbor >/dev/null || fail 'harbor is not installed or not on PATH'
command -v docker >/dev/null || fail 'docker is not installed or not on PATH'
task=$(cd -- "$task" && pwd)
export QUANT_BENCH_MODEL_DIR=$(cd -- "$model_dir" && pwd)
export QUANT_BENCH_GPU_DEVICE="$gpu"
export QUANT_BENCH_HOST_UID="$(id -u)"
export QUANT_BENCH_HOST_GID="$(id -g)"
export QUANT_BENCH_RUN_DIR="$project_root/jobs/$run_name"
extra=()
if [[ -n "$debug_dir" ]]; then
  [[ -d "$debug_dir" ]] || fail "Debug model directory does not exist: $debug_dir"
  [[ "${task##*/}" == compress-qwen3.6-* || "${task##*/}" == compress-qwen3.8-* ]] || fail 'Debug model support is limited to Qwen3.6 and Qwen3.8 tasks'
  export QUANT_BENCH_DEBUG_MODEL_DIR=$(cd -- "$debug_dir" && pwd)
  extra=(--extra-docker-compose "$project_root/scripts/debug-model-compose.yaml")
fi
mkdir -p "$project_root/jobs"
# mkdir intentionally rejects an existing run, including a previous failed run.
mkdir "$QUANT_BENCH_RUN_DIR" || fail "Choose a new run name; directory already exists or cannot be created: $QUANT_BENCH_RUN_DIR"
mkdir -p "$QUANT_BENCH_RUN_DIR/work/.agent-home" "$QUANT_BENCH_RUN_DIR/submission"
if [[ -n "$debug_dir" ]]; then mkdir "$QUANT_BENCH_RUN_DIR/work/debug-submission"; fi
touch "$QUANT_BENCH_RUN_DIR/.submission.lock"
printf 'Task: %s\nRun directory: %s\n' "$task" "$QUANT_BENCH_RUN_DIR"
exec harbor run -p "$task" -a "$agent" -m "$model" \
  --env docker --force-build \
  --n-attempts 1 --n-concurrent 1 --max-retries 0 \
  --job-name harbor --jobs-dir "$QUANT_BENCH_RUN_DIR" "${extra[@]}"
