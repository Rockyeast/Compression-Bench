#!/bin/bash
set -euo pipefail

# Keep recipe metadata outside the submitted package. Do not overwrite a candidate.
mkdir -p /app/work /app/submission
if [[ -n $(find /app/submission -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo '/app/submission must be empty before running the reference' >&2
    exit 1
fi
reference_run=$(mktemp -d /app/work/reference-12g-XXXXXX)
/opt/quantize/bin/python /solution/quantize_gptq.py \
    --model /input/model --run "$reference_run" \
    --inference /solution/inference.py \
    --calibration /solution/calibration/train.parquet
cp -a "$reference_run/submission/." /app/submission/
