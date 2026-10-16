# Compression and execution: choose a route

Compress the supplied checkpoint using the task's fixed tokenizer and evaluation
rules. Choose freely among established compression methods, new methods and
combinations; scoring measures the submitted model's results.

## Model compression methods

These five routes are peers in this library. They describe what changes in the
model or its stored weights. Routes can overlap: codebook quantization, for
example, combines quantization and encoding. Methods may be combined.

| Route | What changes | Saved-size condition | Runtime condition | Guide |
| --- | --- | --- | --- | --- |
| Quantization | Weight/activation numerical precision | Count packed weights, scales, metadata and retained high-precision tensors | A compatible efficient kernel must consume the format | [quantization](quantization/README.md) |
| Pruning / structure reduction | Nonzeros, neurons, heads, blocks or dimensions | Delete storage or encode sparsity to reduce file size | Physically smaller operations or supported sparse kernels | [pruning](pruning/README.md) |
| Low-rank factorization | A matrix becomes smaller factors | r*(in+out) must beat in*out including precision/metadata | Extra launches and intermediate tensors can offset fewer FLOPs | [low rank](other-compression/low-rank/README.md) |
| Parameter sharing | Multiple consumers reuse stored parameters | Count shared tensors once, plus mappings and corrections | Shared weights may still require separate computation at each use | [sharing](other-compression/parameter-sharing/README.md) |
| Weight encoding / codebooks | Values or patterns use compact codes | Count codes, codebooks, scales, residuals and decoder metadata | Efficient decoding and end-to-end timing establish the runtime benefit | [encoding](other-compression/weight-encoding/README.md) |

## Training and quality recovery

[Distillation / recovery](distillation/README.md) describes how a student is trained or
quality is restored. The student's architecture and representation determine its
size. It can support any compression route above; the teacher must be absent from
deployed inference.

## Computation, runtime memory and execution

[Conditional computation](inference-optimization/dynamic-compute/README.md) changes which operations run for an
input. Evaluate storage savings separately; routing and fallback must implement
the same causal model in every API.
[Cache / runtime-state compression](state-compression/README.md) primarily reduces
working memory or state traffic, and needs its own quality validation. [Kernel, graph and scheduling optimization](inference-optimization/README.md)
primarily improves execution. Measure model-file capacity separately for both routes.

## A bounded experiment

1. Inventory actual tensor shapes, dtypes and bytes. Establish a loadable baseline
   before searching. Measure the final package, including the official tokenizer.
2. Pick a bottleneck and one concrete hypothesis. Estimate possible byte and time
   savings, including decoder, routing, indices, extra factors and recovery code.
3. Separate calibration/training, checkpoint selection and final evaluation texts.
   Reserve Checker and hidden inputs exclusively for evaluation.
4. Use local output-error tests to diagnose implementations. Measure whole-model
   PPL and decode speed on a loadable candidate that meets capacity; use those
   results to assess modest local errors and kernel-level speedups.
5. Evaluate PPL, decode duration, package bytes and actual training/search time.
   Apply the task's quality penalty and hard cap when computing official reward.
   Track any additional research criteria separately. Compare same-model references.
6. Keep the best valid candidate and a reproducible recipe. Remove disposable
   intermediate models only after identifying them as unused; preserve logs,
   configuration, selected weights and any active process inputs.

Try meaningfully different routes before spending the entire budget tweaking one
family. Validate each method's adaptation to the target architecture.
If combining methods, compare the combination against its components under
a matched size and compute budget. Choose the combination from measured bottlenecks.

[The library](README.md) covers quantization, pruning,
distillation, low rank and sharing. The source pack is reference material;
consult the tool table for installed environments and adaptation requirements.

## Runnable tools

See [offline tool entry points](common/README.md) and the
[Tool availability](runtime/tools.md) for available implementations. Author source snapshots
are available inside each method directory under `upstream/`; validate their compatibility with the target model.
