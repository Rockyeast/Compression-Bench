# LLMC / LightCompress

[Author repository](https://github.com/ModelTC/LightCompress). Environment: `/opt/llmc/bin/python`.

Source revision: `86f564ddb1d6548b228c67a10509a4ed7264345c`.

Bundled author source, README and license: [upstream/](upstream/).

```python
from llmc.compression.quantization.quant import IntegerQuantizer
```

Set `PYTHONPATH=/opt/quant-docs/quantization/llmc/upstream` before importing author components.

Bundled author code supports selected architectures. Model registration, multimodal plugins and exporter compatibility need checking for each target; a component check is not validation of all author CLIs.

Offline component check (synthetic inputs only):

```bash
/opt/llmc/bin/python /opt/quant-docs/common/extra_tool_check.py llmc --device cuda
```

This verifies a small component, not full-model quality, speed or task eligibility.

The complete `python -m llmc` entry eagerly imports optional HumanEval and
multimodal evaluation/model plugins (`lmms_eval`, etc.). Those extra stacks are
not included. Use the installed compression components directly or adapt the
author entry; do not assume that the complete CLI is ready.
