# Conditional computation

Conditional computation chooses which operations to run from the current input
and causal history. It changes the computation selected for an input, rather than
necessarily changing how weights or runtime state are stored.

| Route | Concrete change | Costs and checks |
| --- | --- | --- |
| Token/layer/expert routing | Execute a selected subset of available computation | Include router, residual and fallback costs; unused-at-this-step parameters still count toward package capacity |
| Early exit | Produce an output from an intermediate exit head using a defined decision rule | Train/validate the exit heads and rule; include all heads and metadata in package size |
| Sparse attention | Select historical tokens or blocks to attend to | Include selection and indexing overhead; skipping reads does not itself remove allocated KV storage |

Permanent layer or channel removal belongs to [pruning](../../pruning/README.md). Changing
stored KV or recurrent-state representations belongs to
[state compression](../../state-compression/README.md). Making the same operations execute
faster belongs to [inference optimization](../README.md). These
routes can be combined, but report their effects separately.

## Task boundaries

Only optimize permitted text-language components. Preserve visual and other
non-text modality modules unchanged, as required by the task instruction.
Routing decisions must use only available prefix information. Generation,
streaming and token probabilities must implement the same submitted causal model.
For a model with approximate routing, validate that model's quality; do not score
PPL with a different full-compute model while timing a cheaper decode path.

Speculative decoding is prohibited: no draft model, MTP, EAGLE, n-gram speculation
or multi-token verification followed by artificial single-token emission.
Conditional computation does not provide an exception.

## A small experiment

1. Define one decision rule, its available inputs, and the fallback path.
2. Record how often each branch runs on held-out inputs, including difficult and
   long sequences. A rarely used fast branch does not establish overall speedup.
3. Compare probabilities for the same prefixes through prefill and incremental
   decode, accounting for the method's stated numerical tolerance.
4. Measure full-model quality and complete decode latency, including decisions,
   indexing and fallback. Run the official Checker and keep all package bytes
   within the limit even if some weights are conditionally unused.

This guide supplies design and validation guidance, not a ready-to-run routing
implementation. Inspect the installed model/runtime and prototype one compatible
path before integrating the whole model.

[Compression overview](../../overview.md)

## Methods and tools

Entries are alphabetical, not recommendations or a ranking. Read each method's
scope, dependencies and adaptation limits before choosing an implementation.

- [ctxfold](ctxfold/README.md)
- [minference](minference/README.md)
- [packrat](packrat/README.md)

See [shared helpers](../../common/README.md) and
[environment availability](../../runtime/tools.md).
