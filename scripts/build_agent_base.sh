#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
reference_args=()
if [[ -n "${QUANTBENCH_REFERENCE_ARCHIVE_DIR:-}" ]]; then
  reference_args+=(--archive-dir "$QUANTBENCH_REFERENCE_ARCHIVE_DIR")
fi
python3 "$project_root/scripts/prepare_references.py" "${reference_args[@]}"
# Build the shared inference stack first, then add Agent-only creation tools.
docker build -t quantbench-runtime:local "$@" "$project_root/shared/runtime"
exec docker build -t quantbench-agent-base:local "$@" "$project_root/shared/agent"
