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
| Starter inference backend: vLLM 0.28.0 | `python3` |
| Shared inference extensions: GGUF plugin, Humming 0.1.13, math-verify 0.9.0 | Same base runtime in Agent, Checker and Verifier |
| Original checkpoint, read-only | `/input/model/` |
| Official tokenizer for calibration/local inference, read-only | `/input/model/` |
| Scratch scripts, logs and caches (temporary; not automatically archived) | `/app/work/` |
| Final model and inference.py | `/app/submission/` |
| Public tools | `/app/model_inventory.py`, `/app/check_submission.py` |

Tool interpreters, entrypoints and compatibility limits are listed in the
[tool environment table](tools.md). Use the interpreter specified there when
running a tool; environments do not switch automatically. Exported checkpoints
must be supported by the final inference runtime.

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

## Text-only scope and optional non-text modules

Current text-only tasks allow removal of visual and other non-text modality encoders, projectors/adapters and their modality-only configuration. Keep the submitted configuration consistent with your loader and retain everything needed for genuine text generation and token probabilities. Any non-text modules retained must remain unchanged and count toward package capacity. The starter uses `language_model_only=True` to skip non-text execution; removing stored tensors still requires a consistent export and loader. Follow the selected task instruction for its exact scope and MTP rules.
