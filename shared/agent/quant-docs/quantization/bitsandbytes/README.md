# bitsandbytes

[Author repository](https://github.com/bitsandbytes-foundation/bitsandbytes). Environment: `/opt/bitsandbytes/bin/python`.

```python
from bitsandbytes.nn import Linear4bit
```

CUDA NF4/FP4 and INT8 tools. Optional CPU kernels from the Hugging Face Hub are not included. Task execution must remain offline.

Offline component check (synthetic inputs only):

```bash
/opt/bitsandbytes/bin/python /opt/quant-docs/common/extra_tool_check.py bitsandbytes --device cuda
```

This check covers a small component. Validate full-model quality, speed and task eligibility separately.
