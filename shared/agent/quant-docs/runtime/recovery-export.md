# Align training, export and deployment

Let Q(W) denote the actual rounded/clipped weight representation consumed by the
runtime, and Delta a learned correction. These computations differ:

    live branch:       y = Q(W) x + Delta x
    merged deployment: y = Q(W + Delta) x

If the training base is the dequantized Q(W), replace W in the second expression
with that exact base. Quantization is nonlinear, so Q(W+Delta) generally differs
from Q(W)+Delta. A small correction can disappear, change rounding boundaries or
alter shared group/block scales after merging. Preserve the precise base checkpoint.

There are two deployment choices:

1. Keep the correction branch. Training and serving can agree more closely, but
   branch parameters count toward size and extra matrix/launch work counts toward
   decode. A fused branch still needs quality and speed validation.
2. Merge and requantize. Inference can use the ordinary quantized path, but validate
   the final packed artifact. Training with a quantized merged forward can reduce
   mismatch; a straight-through gradient or fake quantizer must reproduce relevant
   rounding, clipping, scales, dtype and layout, not merely say "QAT".

Compare four states on the same independent texts: unchanged student, live trained
student, merged/dequantized student and final runtime-loaded exported student. Also
compare trained-text and held-out losses. This separates ineffective training,
overfitting, merge damage and runtime defects.

Activation quantization, kernel accumulation precision and sequential cache/state
updates may introduce additional differences beyond weight quantization. Record
which are modeled by training. A local fake-quant test does not certify the complete
runtime. Store recipe, source identity, optimizer/data budget and validation choice;
ship only the weights/code required for standalone inference.
