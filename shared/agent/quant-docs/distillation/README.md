# Distillation and recovery training

Distillation is a training objective. Size reduction comes from the student's
structure, representation or sharing, not from calling training "distillation".
The teacher may be the supplied uncompressed checkpoint during training; deployed
inference must load only the submission and must not consult that checkpoint.

## A concrete recovery experiment

1. Initialize the student from a loadable pruned/quantized model. Save its untrained
   quality and exact export recipe as a control.
2. Use allowed training data, a separate checkpoint-selection split and independent
   final quality evaluation. The optional WikiText training material can support
   calibration or a pilot; it is not evidence that a long recovery has enough data.
3. Freeze teacher weights and run it in eval/no-grad mode. Choose student trainables
   explicitly: full weights, selected projections, norms or low-rank adapters.
   Log trainable parameter count, rank where applicable, tokens, steps and optimizer.
4. For teacher/student logits z_t,z_s, soft-target loss can be
   T^2 * KL(softmax(z_t/T) || softmax(z_s/T)); optionally combine it with ordinary
   next-token cross-entropy. The task tokenizer fixes the vocabulary alignment.
   Feature matching is optional and may need a projection when widths differ.
5. Score all valid next-token positions for a dense loss. If sampling positions to
   save memory, record exactly how many and how selected; distinguish that training
   approximation from full held-out evaluation. Chunk vocabulary/time computation
   to control logits memory without silently changing the loss.
6. Record a learning curve and save resumable checkpoints at a few useful intervals.
   Compare trained-text and unseen-text losses. Select on validation data; test the
   selected exported model once on independent final data. Do not select on hidden
   evaluation inputs or normalize loss across differing token counts incorrectly.

This is gradient-based supervised/distillation training, not reinforcement learning.
No RL framework is required. A low-rank branch restricts update capacity; rank 8 is
an experiment parameter, not a default justified for every compression level.

## Diagnose before extending training

| Observation | Next check |
| --- | --- |
| Training loss does not improve | Gradient flow, optimizer, loss direction/alignment, update capacity and learning rate |
| Training improves, held-out does not | Data diversity, overfitting, selection-set size and distribution shift |
| Live student improves, exported student worsens | Merge/rounding/clipping, scale recalculation and runtime mismatch |
| Quality improves but decode slows | Extra branches, non-fused adapters or changed matrix shapes |

Benchmark a short pilot to estimate tokens/second and memory before allocating the
whole task budget. Keep teacher/student compute, checkpoint I/O and validation in
the budget; papers' full recovery schedules may be much larger than an Agent run.
Reusing a checkpoint requires optimizer, scheduler, RNG and data-position state,
not merely loading model weights.

See [training/export alignment](../runtime/recovery-export.md) before choosing
how trained updates or adapters are saved. A published recovery recipe is not
a prevalidated adapter for every model.

## Methods and tools

Entries are alphabetical, not recommendations or a ranking. Read each method's
scope, dependencies and adaptation limits before choosing an implementation.

- [alpaca-cot](alpaca-cot/README.md)
- [bmcook](bmcook/README.md)
- [easyllm](easyllm/README.md)
- [knowledge-fidelity](knowledge-fidelity/README.md)
- [lit-llama](lit-llama/README.md)
- [litgpt](litgpt/README.md)
- [llamafactory](llamafactory/README.md)
- [llamatuner](llamatuner/README.md)
- [megatron](megatron/README.md)
- [minitron](minitron/README.md)
- [unsloth](unsloth/README.md)

See [shared helpers](../common/README.md) and
[environment availability](../runtime/tools.md).
See the [offline paper catalogue](papers/README.md) for further references.
