FROM vllm/vllm-openai:v0.28.0

# Keep vLLM fixed; override its Humming pin for the upstream SM90 multicast fix.
# Validated with vLLM 0.28.0; --no-deps prevents unrelated package upgrades.
RUN python3 -m pip install --no-cache-dir --no-deps humming-kernels==0.1.13 \
    && python3 -c "from importlib.metadata import version; assert version('vllm') == '0.28.0'; assert version('humming-kernels') == '0.1.13'"

ENTRYPOINT []
COPY test.sh evaluate.py tokenizer_assets.py inference_runtime.py decode_benchmark.py /tests/
COPY ppl_benchmark.py /tests/
COPY evaluator-assets /tests/evaluator-assets
RUN chmod 0555 /tests/test.sh /tests/evaluate.py
WORKDIR /tests
