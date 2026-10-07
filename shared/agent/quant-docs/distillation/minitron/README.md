# Minitron

Structured pruning with knowledge distillation. Read [pruning](../../pruning/README.md),
[distillation](../README.md) and [export alignment](../../runtime/recovery-export.md).
[Paper](https://arxiv.org/abs/2407.14679) · [text](paper.txt) · [PDF](paper.pdf).
Paper recovery budgets are not guarantees of recovery within this task's budget.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../references/papers.json)

Paper reference only; [generic recovery](../../common/recover.py) is not a complete Minitron reproduction.

[Tool availability](../../runtime/tools.md)

## Available tools and limitations

The inspected [NVlabs/Minitron repository](https://github.com/NVlabs/Minitron)
contains model cards and links; its practical pruning/distillation flow points to
NeMo/ModelOpt and a separate training stack, which is not included in this image.
Use `common/shrink_mlp.py` and `common/recover.py` in `/opt/quantize` for basic
structural compression/recovery. These tools are not a Minitron reproduction.
The full author workflow requires additional model conversion, training and
export integration.
