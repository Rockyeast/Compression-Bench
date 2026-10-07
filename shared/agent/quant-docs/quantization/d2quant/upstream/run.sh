#!/usr/bin/env bash
# Example D²Quant commands. Set MODEL to a local Hugging Face checkpoint.
set -euo pipefail

MODEL=/path/to/Meta-Llama-3-8B
GPU=0

# Full D²Quant: GPTAQ + DSQ + DAC, followed by WikiText-2 / C4 evaluation.
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --model "$MODEL" \
    --bits 2 \
    --group-size 128 \
    --dsq \
    --dac \
    --ppl-eval

# Ablations:
#   --no-dsq            disable Dual-Scale Quantization
#   --no-dac            disable Deviation-Aware Correction
#   --no-dsq --no-dac   rotated GPTAQ baseline
#   --no-rotate         disable the QuaRot transformation
