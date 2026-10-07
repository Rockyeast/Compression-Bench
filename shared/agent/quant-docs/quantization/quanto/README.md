# Optimum Quanto

[Author repository](https://github.com/huggingface/optimum-quanto). Environment: `/opt/quanto/bin/python`.

```python
from optimum.quanto import quantize, freeze, qint8
```

Generic PyTorch quantization API. Save and reload using a compatible runtime; this is not automatically a vLLM checkpoint.

Offline component check (synthetic inputs only):

```bash
/opt/quanto/bin/python /opt/quant-docs/common/extra_tool_check.py quanto --device cuda
```

This verifies a small component, not full-model quality, speed or task eligibility.
