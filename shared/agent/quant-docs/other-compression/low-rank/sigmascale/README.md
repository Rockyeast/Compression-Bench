# SigmaScale

[Author repository](https://github.com/ernlavr/SigmaScale). Environment: `/opt/sigmascale/bin/python`.

Source revision: `8e348712ecb6f1b7eb38c129af9deb49949d2b01`.

Bundled author source, README and license: [upstream/](upstream/).

```python
from utils import low_rank_utils
```

Run from `/opt/quant-docs/other-compression/low-rank/sigmascale/upstream`, or add it to `PYTHONPATH`. Start with `utils.low_rank_utils` (the author entry order) to avoid circular imports.

Author low-rank workflow provides Llama and Qwen3 reconstruction classes; this does not establish support for hybrid GDN architectures. Use local training data and adapt its data loader if needed. Do not run its bundled full requirements.txt: it includes a ROCm-specific package.

Offline component check (synthetic inputs only):

```bash
/opt/sigmascale/bin/python /opt/quant-docs/common/extra_tool_check.py sigmascale --device cuda
```

This verifies a small component, not full-model quality, speed or task eligibility.
