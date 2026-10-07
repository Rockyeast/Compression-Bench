# NVIDIA ModelOpt

[Author repository](https://github.com/NVIDIA/Model-Optimizer). Environment: `/opt/modelopt/bin/python`.

```python
import modelopt.torch.quantization as mtq
```

PTQ/QAT, pruning and distillation APIs are available. TensorRT engines and Megatron/NeMo training stacks are not installed. Model adapters and exported formats must match the task runtime.

Offline component check (synthetic inputs only):

```bash
/opt/modelopt/bin/python /opt/quant-docs/common/extra_tool_check.py modelopt --device cuda
```

This verifies a small component, not full-model quality, speed or task eligibility.
