# Compression reference library

Read the task instruction first. Start with [the overview](overview.md), choose a
route below, then open a method's README. Each method keeps its paper, dedicated
entry (if provided), and bundled author source together. Shared helpers are linked
rather than copied. Installed Python environments remain at their existing /opt paths.

| Direction | What changes |
| --- | --- |
| [Quantization](quantization/README.md) | Numerical representation of weights/activations |
| [Pruning](pruning/README.md) | Nonzeros, neurons, layers or dimensions |
| [Distillation / recovery](distillation/README.md) | Training a smaller model or recovering quality |
| [Other model compression](other-compression/README.md) | Low rank, parameter sharing and weight encoding |
| [Prompt compression](prompt-compression/README.md) | Compact input representations; subject to task rules |
| [Runtime-state compression](state-compression/README.md) | KV/recurrent-state representation |
| [Inference optimization](inference-optimization/README.md) | Execution efficiency and conditional computation |

[Surveys](surveys/README.md) collect overview papers rather than a compression method.

- [Common tools](common/README.md): shared pruning, structural reduction, recovery and matrix helpers.
- [Runtime](runtime/README.md): environment, submission API, export and debugging.
- [References](references/README.md): official documents and source locations.

A paper or source snapshot does not imply a task-ready implementation. The task's
modality-preservation, capacity, quality and streaming rules remain authoritative.

Tool availability is explicit in the [tool availability table](runtime/tools.md#available-tools-and-limitations):
installed components, model adaptation and reference-only sources are separate.

[Expanded offline paper library](external/awesome-llm-compression/README.md): papers from Awesome-LLM-Compression, with PDF/TXT files shared across tasks.
