# Cache and runtime-state compression

Runtime state is information retained while processing one request so later
tokens can reuse the past. It is different from the model's learned weights.
An attention layer commonly retains past keys and values (KV cache); a recurrent
layer retains an updated summary state. These buffers live in runtime memory and
are updated as tokens arrive, rather than being ordinary checkpoint parameters.

## What is compressed

| State | Contents | Candidate changes | Main validation risk |
| --- | --- | --- | --- |
| Attention KV cache | Past token key/value vectors used by subsequent attention | Lower-precision storage with scales; selected retention/eviction; a reduced representation with a matching reader | Missing context, reconstruction cost, or different prefill/decode semantics |
| GDN recurrent state | Per-request, per-head matrices summarizing prior key/value updates | Lower precision, reduced dimensions or other-compression/low-rank/encoded state with compatible updates | Approximation error can accumulate through recurrent reads and writes |
| Other text-model streaming state | For example, a short convolution's retained recent inputs | A compact representation supported by the actual update operator | Incorrect history ordering, reset behavior or dtype conversion |

For a Gated DeltaNet (GDN) layer, the recurrent matrix is updated using the next
token's key/value and gates, then used to produce outputs. Compressing this matrix
changes the request's working memory; it does not mean quantizing the learned
projection weights. Exact state layout and dtype depend on the installed model
and kernel. Inventory them before designing a representation.

A simple storage example: 1,000,000 BF16 state elements require 2,000,000 payload
bytes. INT8 payload requires 1,000,000 bytes, plus scales and any metadata. The
runtime may also require temporary dequantized buffers or a high-precision copy;
count these before claiming a memory reduction. Lower storage precision does not
guarantee lower latency if conversions or unsupported kernels add work.

## Distinguish the effects

- **Package size:** temporary KV/state is not the saved model-weight package.
  Compressing it alone does not satisfy a 12/16/24/32 GiB submission limit. Any
  persistent learned adapters, codebooks or helper code still count in the package.
- **Runtime memory:** measure live and peak allocations, including state, scales,
  workspace, graph buffers and duplicated representations, at fixed request count
  and sequence lengths. Unused allocator reservations are not live tensor bytes.
- **Memory traffic and latency:** measure reads, writes, conversion and update
  work together. Fewer stored bytes may help a bandwidth-limited path, but the
  complete decode measurement decides whether it is faster.
- **Quality:** state approximations may change later token probabilities even
  when checkpoint weights are unchanged. Evaluate the compressed-state path.

Reading only selected historical blocks is [conditional computation](../inference-optimization/dynamic-compute/README.md).
Actually evicting or compacting their storage also changes state retention.
Keeping exactly the same state values while fusing writes or changing layout is
[inference execution optimization](../inference-optimization/README.md); it need not be
lossy state compression.

## Validate the actual deployed path

1. Identify the buffers, shapes, dtypes, lifetimes and update/read sites. Establish
   an uncompressed reference using the same weights, prompts and runtime settings.
2. Implement both writing and reading the representation. Check initialization,
   reset, request isolation, variable sequence lengths, padding and graph replay.
3. Check one update and long update sequences against the reference. Report
   absolute/relative errors and their growth; one-step agreement is insufficient.
4. Use held-out fixed token sequences for teacher-forced incremental decode:
   feed the prescribed next token at each step and score its log probability.
   Compare with the reference on exactly the same prefixes. This tests state
   quality without sampling divergence changing all subsequent inputs.
5. The PPL and generation APIs must use the same submitted model semantics. If
   ordinary prefill bypasses the changed state representation, unchanged prefill
   PPL does not validate it. Provide a matching probability path and incremental
   quality evidence; do not leave this mismatch hidden behind a passing Checker.
6. Measure the full decode path and actual memory footprint, then run the formal
   Checker. Report package bytes, runtime bytes, quality and speed separately.

The task still requires one CPU-ready token per streaming event and prohibits
speculative decoding. State compression applies to the text model; it does not
permit modifying protected visual or other non-text modality modules. It also
does not permit cross-request answer caching or using future-token information.

The mechanisms above require adaptation and validation on the actual model.
The references below do not establish task-ready integration or measured gains.

[Compression overview](../overview.md)

## Methods and tools

Entries are alphabetical, not recommendations or a ranking. Read each method's
scope, dependencies and adaptation limits before choosing an implementation.

- [kvpress](kvpress/README.md)

See [shared helpers](../common/README.md) and
[environment availability](../runtime/tools.md).
See the [offline paper catalogue](papers/README.md) for further references.
