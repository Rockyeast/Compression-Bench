#!/bin/bash

set -euo pipefail

/opt/quantize/bin/python /solution/quantize_fp8.py \
    --model /input/model \
    --submission /app/submission
