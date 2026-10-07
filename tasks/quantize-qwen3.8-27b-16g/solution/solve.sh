#!/bin/bash

set -euo pipefail

/opt/quantize/bin/python /solution/quantize_int4.py \
    --model /input/model \
    --submission /app/submission
