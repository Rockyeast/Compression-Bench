# Shared evaluation inputs

`public/` supplies the Checker; `hidden/` supplies the Verifier. Each role receives
only its own split through a separate Docker build context.

Each model family has fixed `ppl_inputs.json` and `decode_inputs.json` files.
The Gemma 26B and 31B tasks use identical token arrays, so both use `gemma4/`.
The optional Qwen 9B debug profile reuses the `qwen3.8/` public arrays. Task-local
configs retain each model's tokenizer, protocol, sampling provenance and scoring.

`public/math500_probe.json` contains the shared public diagnostic questions.
Each task keeps its capacity policy in `evaluator/model_memory_policy.json`.
Task Dockerfiles copy these sources into the existing `evaluator-assets/` runtime
paths alongside task-local configs and tokenizer assets.
