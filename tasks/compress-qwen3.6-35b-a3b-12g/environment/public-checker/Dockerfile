FROM quantbench-runtime:local

# Build context: task root. Keep runtime paths stable for task-owned assets.
COPY --from=shared-evaluator *.py /checker/
COPY --from=common-assets runtime/raw-reference.json /checker/runtime_config.json
COPY environment/public-checker/evaluator-assets/ /checker/evaluator-assets/
COPY evaluator/model_memory_policy.json /checker/evaluator-assets/
COPY --from=public-assets qwen3.6/ /checker/evaluator-assets/
COPY --from=public-assets math500_probe.json /checker/evaluator-assets/
COPY --from=public-assets qwen3.8/ /checker/evaluator-assets/debug-9b/
COPY --from=common-assets tokenizers/qwen3.6/ /checker/evaluator-assets/tokenizer/
COPY --from=public-assets debug-9b/ /checker/evaluator-assets/debug-9b/
COPY --from=public-assets profiles/with-debug.json /checker/evaluator-assets/evaluation_profiles.json
RUN chmod 0555 /checker/server.py /checker/evaluate_public.py

ENTRYPOINT ["python3", "/checker/server.py"]

RUN python3 /opt/quantbench-runtime/verify_runtime.py
