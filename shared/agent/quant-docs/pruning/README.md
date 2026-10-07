# Pruning and physical structure reduction

Pruning can remove individual weights, complete MLP channels, attention heads or
whole blocks. These choices have different file-size and execution consequences.

## Sparse weights

Weight selection can use magnitude, activation statistics or reconstruction error,
with or without compensation of the remaining weights. Choose and compare scoring
rules under matched sparsity and calibration budgets. Published quality results
do not guarantee a sparse speedup on the task GPU.
A dense tensor full of zeros has the same shape and byte count. Export a supported
sparse representation with indices/metadata, or use a supported N:M kernel, and
measure its actual storage and execution. Do not assume any 50%-sparse tensor
qualifies for 2:4 acceleration.

## Channels and dimensions

Removing a channel requires matching changes in every producer and consumer,
including biases, normalization parameters, quantization scales and packed groups.
For gated projections, preserve the correspondence between parallel branches and
the downstream projection. For attention, respect head/group relationships.

Use physically smaller tensors rather than full-size masks. Choose retained widths
from actual kernel constraints. Compare scoring rules under matched size and
recovery budgets; local reconstruction error does not establish final quality.

## Blocks, attention and hybrid models

Whole-block removal saves repeated computation, but requires reindexing layers,
config entries and cache/state ownership. Hybrid Attention/GDN blocks have different
state and projection layouts; do not identify them only by layer number.

Q/K/V head pruning must respect grouped-query relationships and the output projection.
GDN pruning additionally changes recurrent-state dimensions and head/group mappings.
Global hidden-dimension pruning affects residual paths, norms, embeddings, output
heads and many projections. Audit actual model code and exported config first.

## Minimal acceptance sequence

- Save a physically reduced candidate and reload it in a fresh process.
- Verify tensor/config dimensions, causal prefill and multi-step cached decode.
- Count the complete package; evaluate full-model PPL on held-out text.
- If recovery is used, evaluate both before recovery and after final export.
- Time actual decode using the submitted runtime. Removing channels does not
  imply a proportional speedup if a slower matrix kernel is selected.

## Methods and tools

Entries are alphabetical, not recommendations or a ranking. Read each method's
scope, dependencies and adaptation limits before choosing an implementation.

- [minitron](../distillation/minitron/README.md)
- [shortgpt](shortgpt/README.md)
- [slicegpt](slicegpt/README.md)
- [sparsegpt](sparsegpt/README.md)
- [sparseml](sparseml/README.md)
- [wanda](wanda/README.md)

See [shared helpers](../common/README.md) and
[environment availability](../runtime/tools.md).
See the [offline paper catalogue](papers/README.md) for further references.
