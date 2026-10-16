# SliceGPT / Transformer Compression

[Author repository](https://github.com/microsoft/TransformerCompression). Environment: `/opt/slicegpt/bin/python`.

Source revision: `183cd92d8469bf3f1e979585a08dc382692bfbed`.

Bundled author source, README and license: [upstream/](upstream/).

```python
from slicegpt import rotate_and_slice
```

CLI: `/opt/slicegpt/bin/python /opt/quant-docs/pruning/slicegpt/upstream/experiments/run_slicegpt.py --help`.

Author adapters cover Llama, OPT, Phi-2 and Phi-3. New architectures require an adapter. Sliced weights need matching reconstruction/inference code.

Offline component check (synthetic inputs only):

```bash
/opt/slicegpt/bin/python /opt/quant-docs/common/extra_tool_check.py slicegpt --device cuda
```

This check covers a small component. Validate full-model quality, speed and task eligibility separately.
