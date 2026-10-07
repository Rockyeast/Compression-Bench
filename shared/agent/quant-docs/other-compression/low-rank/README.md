# Low-rank factorization

Replace W[out,in] with A[out,r] @ B[r,in]. The stored parameter count changes from
out*in to r*(out+in), before dtype, scales, biases or residual corrections. Thus an
FP16 low-rank representation does not automatically beat an INT4 dense matrix in bytes.
Factor selection may use weights alone, activation statistics or output reconstruction
objectives. Compare with a simple truncated-SVD control under matched rank and budget.

Inference must compute (x @ B.T) @ A.T or a compatible fused operation. Reconstructing
W in full before every token generally defeats the compute objective; reconstructing
once may save disk but not resident memory or steady-state compute. Include both
matrix launches, intermediate traffic and any correction branch in timing. Reload
using the actual exported format and check final PPL, not just matrix MSE.

## Tools and combinations

Choose factor shapes and export/runtime support for the actual model. Source
availability alone does not establish a complete model adapter.

Low-rank factorization is a peer route to [quantization](../../quantization/README.md)
and [pruning](../../pruning/README.md). Factors may also be quantized, and
[distillation/recovery](../../distillation/README.md) may restore quality after factorization.
Count all factors, scales and correction branches in the final package.

## Fair comparisons

Compare against plain SVD and an unfactorized control under matched package-size,
quality and recovery budgets. Keep the runtime fixed where possible; do not
attribute a separate execution optimization to the factorization method.

## Methods and tools

Entries are alphabetical, not recommendations or a ranking. Read each method's
scope, dependencies and adaptation limits before choosing an implementation.

- [sigmascale](sigmascale/README.md)
- [svdllm](svdllm/README.md)

See [shared helpers](../../common/README.md) and
[environment availability](../../runtime/tools.md).
