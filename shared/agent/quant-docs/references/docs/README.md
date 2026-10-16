# Official document snapshots

| Entry point | Usage guide | Concrete example |
| --- | --- | --- |
| `model_free_ptq` | [Official entry-point guide](llmcompressor-model-free.md) | [Local FP8_BLOCK example](llmcompressor-local-examples.md#fp8-example) |
| `oneshot` | [General usage guide](llmcompressor-local-examples.md#oneshot) | [Official GPTQ tutorial](llmcompressor-gptq.md), [local GPTQ example](llmcompressor-local-examples.md#gptq-code) |
| Custom operations | [Official vLLM CustomOp document](vllm-custom-op.md) | [Task integration contract](../../runtime/custom-encoding.md) |

The quantization library documents target LLM Compressor 0.13.0; the general
usage guide and local examples are task-provided notes.
Also available: [vLLM 0.28.0 FP8 guide](vllm-fp8.md) for schemes and runtime loading.

Copyright belongs to the respective LLM Compressor and vLLM contributors.
Both repositories distribute these sources under Apache-2.0; the common license
text is included as [LICENSE-APACHE-2.0.txt](LICENSE-APACHE-2.0.txt).
See also [NOTICE](llmcompressor-NOTICE.txt).
Local quantization examples are in the [local API examples](llmcompressor-local-examples.md).

## Read before using an upstream example

- Do not run the upstream pip-install, git-clone or remote-model/dataset commands
  in the offline task. Use the installed interpreters and provisioned local files.
- Adapt the upstream Kimi/Llama recipes, layer selections and sample counts
  to the task model's architecture and runtime.
- Measure task quality and speed with its Checker.
- Export compressed weights and check the saved files and configuration.
- Choose calibration quantity to balance quality and cost. Minimal smoke examples
  use a few samples to exercise the API; evaluate quality on the exported model.
