FROM vllm/vllm-openai:v0.28.0

# Shared inference dependencies for every Agent, Checker and Verifier.
ARG GGUF_PLUGIN_REV=e2b8ad532b8b5ea175100202c30430c1d2b5e6a8
ARG CUDA_ARCH_LIST
ENV MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}
ENTRYPOINT []
RUN python3 -c 'import os,re; arch=os.environ["TORCH_CUDA_ARCH_LIST"]; assert re.fullmatch(r"[0-9]+\.[0-9]+", arch), "Set --build-arg CUDA_ARCH_LIST to one target GPU compute capability (major.minor)"'
RUN if ! command -v git >/dev/null; then apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*; fi
RUN python3 -m pip install --no-cache-dir --no-deps humming-kernels==0.1.13
# Preserve the inference stack while resolving the existing GGUF dependencies.
RUN python3 -c "from importlib.metadata import version; print('\n'.join(n+'=='+version(n) for n in ['vllm','torch','transformers','triton','humming-kernels']))" > /tmp/runtime-constraints.txt
RUN python3 -m pip install --no-cache-dir -c /tmp/runtime-constraints.txt 'gguf>=0.17.0' 'huggingface_hub>=1.26.0' 'math-verify[antlr4_13_2]==0.9.0'
RUN git clone https://github.com/vllm-project/vllm-gguf-plugin /tmp/vllm-gguf-plugin \
    && cd /tmp/vllm-gguf-plugin && git checkout "$GGUF_PLUGIN_REV" \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && rm -rf /tmp/vllm-gguf-plugin
COPY patch_gguf_text_architecture.py operator-platform-ops.py verify_runtime.py /opt/quantbench-runtime/
RUN python3 /opt/quantbench-runtime/patch_gguf_text_architecture.py \
    && python3 -c "import importlib.util, pathlib, shutil; p=pathlib.Path(importlib.util.find_spec('vllm_gguf_plugin').origin).parent/'ops.py'; shutil.copyfile('/opt/quantbench-runtime/operator-platform-ops.py',p)" \
    && python3 /opt/quantbench-runtime/verify_runtime.py --record

# Shared task tools; task images supply their own evaluation code and inputs.
COPY --chmod=0555 task_tools/agent/*.py /app/
COPY --chmod=0555 task_tools/checker/*.py /checker/
