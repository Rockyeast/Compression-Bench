# torchao

[Author repository](https://github.com/pytorch/ao). Environment: `/opt/torchao/bin/python`.

```python
from torchao.quantization import quantize_, Int8WeightOnlyConfig
```

Use supported hardware and dtype recipes. Optional MSLK/CUTLASS backends are not installed; test the selected recipe before compressing a full model.

Offline component check (synthetic inputs only):

```bash
/opt/torchao/bin/python /opt/quant-docs/common/extra_tool_check.py torchao --device cuda
```

This verifies a small component, not full-model quality, speed or task eligibility.
