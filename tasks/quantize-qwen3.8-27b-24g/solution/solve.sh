#!/bin/bash

set -euo pipefail

/opt/quantize/bin/python /solution/quantize_int6.py \
    --model /input/model \
    --submission /app/submission
