# HQQ

[Author repository](https://github.com/mobiusml/hqq). Environment: `/opt/hqq/bin/python`.

```python
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
```

Core PyTorch backend is provided. Optional GemLite/Marlin backends and full-model export compatibility need separate integration.

Offline component check (synthetic inputs only):

```bash
/opt/hqq/bin/python /opt/quant-docs/common/extra_tool_check.py hqq --device cuda
```

This verifies a small component, not full-model quality, speed or task eligibility.
