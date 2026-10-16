FROM quantbench-runtime:local

ENTRYPOINT []
# Build context: task root. The hidden image receives only hidden assets.
COPY --from=shared-evaluator *.py test.sh /tests/
COPY --from=common-assets runtime/raw-reference.json /tests/runtime_config.json
COPY tests/evaluator-assets/ /tests/evaluator-assets/
COPY evaluator/model_memory_policy.json /tests/evaluator-assets/
COPY --from=hidden-assets gemma4/ /tests/evaluator-assets/
COPY --from=common-assets tokenizers/gemma4/ /tests/evaluator-assets/tokenizer/
RUN chmod 0555 /tests/test.sh /tests/evaluate.py
WORKDIR /tests

RUN python3 /opt/quantbench-runtime/verify_runtime.py
