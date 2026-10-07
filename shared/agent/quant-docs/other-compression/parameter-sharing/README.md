# Parameter sharing

Shared layers/tensors reuse one stored parameter set. Similar dimensions alone do
not imply compatible functions. Retain an explicit mapping and ensure the serializer
does not write a duplicate per consumer. Safetensors storage is not automatic alias
preservation; a custom loader may need to reconstruct sharing.
Sharing learned during original training and tying a pretrained model's parameters
are different settings. Validate whether recovery is needed and measure quality loss.

## Compression and execution

Sharing reduces the number of independently stored parameters. Each consumer may
still perform its own matrix multiplication, so fewer stored bytes do not imply
fewer FLOPs or faster decode. Count maps, layer-specific corrections and retained
unshared weights. Validate that reload preserves the intended shared tensors.

There is no bundled general-purpose sharing adapter. Implement and validate the
mapping, export and loader for the actual architecture; the
[custom encoding contract](../../runtime/custom-encoding.md) explains storage/runtime
responsibilities. Use [distillation/recovery](../../distillation/README.md) if needed.

## Fair comparisons

Keep a no-sharing control with the same inference implementation. Compare a
proposed pairing rule against simple sharing/selection with the same number of
shared parameters and the same recovery budget. Check full-model quality and
actual decode performance, not only pairwise weight similarity.

## Methods and tools

Entries are alphabetical, not recommendations or a ranking. Read each method's
scope, dependencies and adaptation limits before choosing an implementation.

- [albert](albert/README.md)

See [shared helpers](../../common/README.md) and
[environment availability](../../runtime/tools.md).
