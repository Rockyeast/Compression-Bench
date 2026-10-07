FROM vllm/vllm-openai:v0.28.0

# Keep vLLM fixed; override its Humming pin for the upstream SM90 multicast fix.
# Validated with vLLM 0.28.0; --no-deps prevents unrelated package upgrades.
RUN python3 -m pip install --no-cache-dir --no-deps humming-kernels==0.1.13 \
    && python3 -c "from importlib.metadata import version; assert version('vllm') == '0.28.0'; assert version('humming-kernels') == '0.1.13'"

RUN python3 -m pip install --no-cache-dir "math-verify[antlr4_13_2]==0.9.0" \
    && python3 -c "from importlib.metadata import version; assert version('vllm') == '0.28.0'; assert version('humming-kernels') == '0.1.13'; assert version('math-verify') == '0.9.0'"

COPY quality_check.py /checker/
COPY server.py evaluate_public.py evaluate.py tokenizer_assets.py inference_runtime.py /checker/
COPY decode_benchmark.py /checker/
COPY ppl_benchmark.py /checker/
COPY evaluator-assets/ /checker/evaluator-assets/
RUN chmod 0555 /checker/server.py /checker/evaluate_public.py

ENTRYPOINT ["python3", "/checker/server.py"]
