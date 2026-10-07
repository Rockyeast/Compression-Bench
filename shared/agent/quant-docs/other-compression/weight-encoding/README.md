# Weight encoding and codebooks

Vector/codebook representations store indices plus a codebook and possibly scales
or residuals. Count all of them. Verify that a suitable decoder/kernel exists,
or implement it within the submitted runtime. A compact archive that cannot
execute efficiently is a size result, not a speed result.

## Scope and combinations

This route concerns the storage and decoding format. Lossless encoding preserves
values; lossy codebooks also change their numerical representation and can overlap
with [quantization](../../quantization/README.md). These are practical reading categories,
not mutually exclusive mathematical classes. [Parameter sharing](../parameter-sharing/README.md)
focuses on reusing parameters across consumers; encoding focuses on representing
the stored values or patterns compactly.

Follow the [custom encoding interface](../../runtime/custom-encoding.md) for the
loader and runtime implementation. There is no bundled general-purpose encoding
adapter. Decompressing once at load time may reduce package size while leaving
resident memory and steady-state compute unchanged.

## Fair comparisons

Count indices, dictionaries, scales, residuals, decoder code and metadata. Compare
against a simple packed representation under the same quality budget. Measure
load cost separately from steady-state decode. Use the actual exported model for
PPL and speed checks; a smaller archive alone does not establish faster inference.

## Methods and tools

Entries are alphabetical, not recommendations or a ranking. Read each method's
scope, dependencies and adaptation limits before choosing an implementation.

- [qtip](../../quantization/qtip/README.md)

See [shared helpers](../../common/README.md) and
[environment availability](../../runtime/tools.md).
