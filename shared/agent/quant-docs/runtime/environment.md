# Environment and public tools

Copy and adapt `/app/inference.py`. Public/hidden evaluation loads the submitted
engine for generation, token log probabilities and streamed decoding; the task
computes PPL, timestamps token arrival and computes reward.
Public and hidden checks use separate inputs. Results remain provisional until
independent code/log review. See [the inference API](inference-api.md) for method
contracts and review status. The task instruction defines scoring, submission
rules and budgets.

| Purpose | Interpreter / location |
| --- | --- |
| Quantization: LLM Compressor 0.13.0 | `/opt/quantize/bin/python` |
| Sensitivity-guided allocation: AutoRound 0.14.2 | `/opt/autoround/bin/python`; [example and guide](../quantization/autoround/README.md) |
| Asymmetric calibration: GPTAQ via GPTQModel 7.5.0 | `/opt/gptaq/bin/python`; [example and guide](../quantization/gptaq/README.md) |
| SINQ quantization | `/opt/sinq/bin/python`; [usage and boundaries](../quantization/sinq/README.md) |
| SVD-LLM low rank | `/opt/svdllm/bin/python`; [usage and boundaries](../other-compression/low-rank/svdllm/README.md) |
| Starter inference backend: vLLM 0.28.0 | `python3` |
| Original checkpoint, read-only | `/input/model/` |
| Official tokenizer for calibration/local inference, read-only | `/input/model/` |
| Scratch scripts, logs and caches (temporary; not automatically archived) | `/app/work/` |
| Final model and inference.py | `/app/submission/` |
| Public tools | `/app/model_inventory.py`, `/app/check_submission.py` |

The quantization venv inherits base packages and installs its own dependencies.
This reduces package conflicts. Passing a model
between interpreters requires a checkpoint format supported by the final runtime.

The task supplies the tokenizer. For calibration and local tests, use its
read-only files under `/input/model`. The formal interface is Python `load()`
plus the engine methods. The starter's optional `--tokenizer-dir` flag supports
local CLI tests.
Evaluation tokenizes inputs with the official tokenizer and passes token IDs to
your engine. Load weights from the provided `model_dir`.
The host wrapper places official tokenizer assets in the final archive, and the
Checker counts those files toward the capacity limit even if you omit them.

When provisioned, `/input/debug-model` is an optional smaller debugging model.
Check that its config exists and follow the instruction's `--debug-9b` workflow.
Validate the final solution separately on the formal model.

Inspect APIs without loading a model:

```bash
/opt/quantize/bin/python -c 'import inspect; from llmcompressor import model_free_ptq, oneshot; print(inspect.signature(model_free_ptq)); print(inspect.signature(oneshot))'
python3 -c 'import importlib.metadata as m; print(m.version("vllm")); print(m.version("safetensors"))'
```

Inspect tensor metadata on CPU:

```bash
python3 /app/model_inventory.py /input/model --json /app/work/model-inventory.json
```

Rows group identical shapes/dtypes and replace numeric name components with `*`.
The JSON also preserves individual tensor names. Scanning reads tensor metadata
on CPU. Size estimates count stored tensors, including repeated storage; packed
integer tensors are counted according to their stored representation.

Read the [quantization guide](../quantization/README.md) for quantization and export.
See [debugging](debugging.md) for compatibility, source lookup and failed checks.

## Non-text modality preservation

The input is the complete checkpoint. Visual and other non-text modality weights and their associated configuration must be retained unchanged in the submission; optimizing these modules is outside task scope. They count toward package capacity. The starter uses `language_model_only=True` to skip their execution during text evaluation; this is not permission to remove their weights. MTP remains governed separately by the task instruction.
