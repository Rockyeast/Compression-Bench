# syntax=docker/dockerfile:1.14
# scripts/build_agent_base.sh prepares and verifies the pinned offline references first.
# Only installation sources enter this intermediate stage. Guide/paper edits
# leave the expensive tool-install stages cached; sources are mounted read-only.
FROM scratch AS tool-sources
COPY quant-docs/common/upstream/fast-hadamard-transform/ /opt/quant-docs/common/upstream/fast-hadamard-transform/
COPY quant-docs/quantization/qtip/upstream/qtip-kernels/ /opt/quant-docs/quantization/qtip/upstream/qtip-kernels/
COPY quant-docs/quantization/llmc/upstream/ /opt/quant-docs/quantization/llmc/upstream/
COPY quant-docs/pruning/slicegpt/upstream/ /opt/quant-docs/pruning/slicegpt/upstream/
COPY quant-docs/other-compression/low-rank/sigmascale/upstream/ /opt/quant-docs/other-compression/low-rank/sigmascale/upstream/
COPY quant-docs/quantization/awq/upstream/awq/ /opt/quant-docs/quantization/awq/upstream/awq/
COPY quant-docs/quantization/bitorch/upstream/ /opt/quant-docs/quantization/bitorch/upstream/
COPY quant-docs/quantization/autofp8/upstream/ /opt/quant-docs/quantization/autofp8/upstream/
COPY quant-docs/distillation/lit-llama/upstream/ /opt/quant-docs/distillation/lit-llama/upstream/
COPY setup/ /tmp/tool-setup/

FROM quantbench-runtime:local
# Build-time only: avoid persisting regenerable Python bytecode in tool layers.
ARG PYTHONDONTWRITEBYTECODE=1

# Git is supplied by the shared runtime for the pinned GPTQ source installation.
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    python3 -m venv --system-site-packages /opt/quantize \
    && /opt/quantize/bin/pip install --no-compile --no-cache-dir -r /tmp/requirements.txt

# Keep completed Python environments cached while compiling per-tool CUDA extensions.
RUN --mount=type=bind,from=tool-sources,source=/opt/quant-docs,target=/opt/quant-docs \
    --mount=type=bind,from=tool-sources,source=/tmp/tool-setup,target=/tmp/tool-setup \
    --mount=type=cache,id=quantbench-agent-pip,target=/root/.cache/pip \
    python3 /tmp/tool-setup/install.py --skip-native --tool-group python
RUN --mount=type=bind,from=tool-sources,source=/opt/quant-docs,target=/opt/quant-docs \
    --mount=type=bind,from=tool-sources,source=/tmp/tool-setup,target=/tmp/tool-setup \
    --mount=type=cache,id=quantbench-agent-pip,target=/root/.cache/pip \
    python3 /tmp/tool-setup/install.py --skip-native --tool-group native
RUN --mount=type=bind,from=tool-sources,source=/opt/quant-docs,target=/opt/quant-docs \
    --mount=type=bind,from=tool-sources,source=/tmp/tool-setup,target=/tmp/tool-setup \
    --mount=type=cache,id=quantbench-agent-pip,target=/root/.cache/pip \
    python3 /tmp/tool-setup/install.py --native-only

# Set permissions while copying, avoiding a second whole-library chmod layer.
COPY --chmod=a=rX quant-docs/ /opt/quant-docs/
# Fail at build time if the bundled non-quantization entry points cannot import.
RUN /opt/quantize/bin/python -c "from llmcompressor.modifiers.pruning import WandaPruningModifier, SparseGPTModifier" \
    && /opt/quantize/bin/python /opt/quant-docs/common/recover.py --help >/dev/null \
    && /opt/quantize/bin/python /opt/quant-docs/common/shrink_mlp.py --help >/dev/null \
    && /opt/quantize/bin/python /opt/quant-docs/common/matrix.py --help >/dev/null
RUN /opt/gptq/bin/python /opt/quant-docs/quantization/gptq/quantize.py --help >/dev/null
# CPU author checks do not require a GPU during image build.
RUN /opt/svdllm/bin/python /opt/quant-docs/common/author_check.py svdllm \
    && /opt/sinq/bin/python /opt/quant-docs/common/author_check.py sinq \
    && /opt/sinq/bin/python /opt/quant-docs/quantization/sinq/quantize.py --help >/dev/null
RUN /opt/spinquant/bin/python /opt/quant-docs/common/research_check.py spinquant \
    && /opt/d2quant/bin/python /opt/quant-docs/common/research_check.py d2quant \
    && /opt/qtip/bin/python /opt/quant-docs/common/research_check.py qtip

# Generic long-command runner available to every Agent, independent of host adapters.
COPY --chmod=644 run_managed_job.py /app/run_managed_job.py

RUN python3 /opt/quantbench-runtime/verify_runtime.py
