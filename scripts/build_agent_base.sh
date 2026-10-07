#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Docker reuses unchanged dependency layers; rebuild after editing shared assets.
exec docker build -t quantbench-agent-base:20260930-shared-v1 "$@" "$project_root/shared/agent"
