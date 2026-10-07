# Inference execution optimization

A smaller model is not necessarily faster; faster execution need not change weights.
Use the same model/precision and workload to isolate execution improvements, then
separately evaluate model compression. Record the actual selected backend, versions,
hardware, active flags and cold versus warm costs.

| Layer | Candidates | What to measure |
| --- | --- | --- |
| Operators/kernels | Fusion, tiling, vectorization, supported matrix-kernel dispatch | Numerical error, representative shapes, full operation chain including conversion |
| Graph/runtime | CUDA Graph, launch reduction, allocation reuse, synchronization removal | CPU/GPU timeline, capture/replay correctness and buffer lifetimes |
| Attention/cache/state | Cache layout, direct writes, less gather/copy, sparse attention | Selection overhead, read/write cost, multi-step state correctness |
| Serving/scheduling | Continuous batching, chunked prefill, admission/scheduling policy | Queue delay, TTFT, inter-token latency, throughput and starvation |
| Parallelism/communication | TP/PP, collectives, overlap | End-to-end latency including communication and synchronization |
| Decoding/control | Sampling implementation, causal conditional execution | Full probability/streaming semantics; obey the speculation ban |

For choosing which operations run, see [conditional computation](dynamic-compute/README.md).
For changing the stored KV/recurrent-state representation, see
[state compression](../state-compression/README.md). These are separate from making the
same operations and state values execute more efficiently.

## Scope of this benchmark

The formal task allocates one GPU and measures concurrency 1, with fixed input/output
lengths and streaming timing. Multi-GPU execution, a larger batch, a shorter output,
request caching or changed timing boundaries are not valid ways to improve this
score. Serving-level scheduling is a broader research direction; it may have no
benefit on this fixed workload. CPU work and data movement required during timed
decode still count. Do not emit already-computed tokens as a fake fast stream.

## Profile, implement, verify

1. Load and warm up the reference once where possible. Reuse a running engine only
   when model loading, cache reset, packed weights and captured pointers remain valid.
   A different checkpoint is not automatically safe to hot-swap into a captured graph.
2. Separate loading/JIT/capture, prefill and decode. Profile representative token
   counts and sequence lengths. A profiled kernel-time sum is not automatically
   elapsed wall time; overlapping kernels and CPU gaps matter.
3. Inspect the installed backend before writing a replacement. Include supported
   built-in optimizations in the reference. Enabling an existing flag is configuration
   tuning, not a newly implemented kernel.
4. Test the complete replaced path with the same inputs, weights and dtype. Cover
   graph replay, multiple calls, noncontiguous strides, padding, aliasing and output
   buffer lifetime. Independent outputs must not accidentally reuse writable storage.
5. Run full-model quality and same-condition decode. A kernel used only in decode
   needs incremental-decode quality checks; unchanged prefill PPL alone is insufficient.
6. Repeat paired/interleaved measurements, retain all samples, report variation and
   cold-start costs separately. Use a combined all-off/all-on comparison for total
   benefit; do not add percentages from unrelated experiments. Ablations identify
   which changes help and whether their effects overlap.

For fixed work, latency reduction is 1 - t_new/t_ref; throughput gain is
 t_ref/t_new - 1. They are different percentages. A 20% local reduction in a component
occupying 10% of nonoverlapping total time has an ideal 2% total-time reduction,
not 20%. Confirm any estimate with end-to-end measurements.

[Inference API](../runtime/inference-api.md) and [custom encoding/kernel guide](../runtime/custom-encoding.md)
define the deployment interface. Consult installed vLLM/FlashInfer/Triton source
for version-specific capabilities; [official custom-op reference](../references/docs/vllm-custom-op.md)
is bundled. This guide does not claim any particular kernel is optimal.

## Methods and tools

Entries are alphabetical, not recommendations or a ranking. Read each method's
scope, dependencies and adaptation limits before choosing an implementation.

- [lmcache](lmcache/README.md)

See [shared helpers](../common/README.md) and
[environment availability](../runtime/tools.md).
