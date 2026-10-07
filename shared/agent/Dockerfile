FROM vllm/vllm-openai:v0.28.0

# Keep vLLM fixed; override its Humming pin for the upstream SM90 multicast fix.
# Validated with vLLM 0.28.0; --no-deps prevents unrelated package upgrades.
RUN python3 -m pip install --no-cache-dir --no-deps humming-kernels==0.1.13 \
    && python3 -c "from importlib.metadata import version; assert version('vllm') == '0.28.0'; assert version('humming-kernels') == '0.1.13'"

ENTRYPOINT []

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv --system-site-packages /opt/quantize \
    && /opt/quantize/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Tool sources are copied before installation; guide edits reuse this layer.
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
RUN python3 /tmp/tool-setup/install.py && rm -rf /tmp/tool-setup

COPY quant-docs/ /opt/quant-docs/
# Fail at build time if the bundled non-quantization entry points cannot import.
RUN /opt/quantize/bin/python -c "from llmcompressor.modifiers.pruning import WandaPruningModifier, SparseGPTModifier" \
    && /opt/quantize/bin/python /opt/quant-docs/common/recover.py --help >/dev/null \
    && /opt/quantize/bin/python /opt/quant-docs/common/shrink_mlp.py --help >/dev/null \
    && /opt/quantize/bin/python /opt/quant-docs/common/matrix.py --help >/dev/null
# CPU author checks do not require a GPU during image build.
RUN /opt/svdllm/bin/python /opt/quant-docs/common/author_check.py svdllm \
    && /opt/sinq/bin/python /opt/quant-docs/common/author_check.py sinq \
    && /opt/sinq/bin/python /opt/quant-docs/quantization/sinq/quantize.py --help >/dev/null
RUN /opt/spinquant/bin/python /opt/quant-docs/common/research_check.py spinquant \
    && /opt/d2quant/bin/python /opt/quant-docs/common/research_check.py d2quant \
    && /opt/qtip/bin/python /opt/quant-docs/common/research_check.py qtip
RUN chmod -R a=rX /opt/quant-docs

# Generic long-command runner available to every Agent, independent of host adapters.
COPY run_managed_job.py /app/run_managed_job.py
RUN chmod 644 /app/run_managed_job.py
