# ShortGPT

[Author repository](https://github.com/icip-cas/ShortGPT), inspected revision
`9a2ee8dfe9e28d0bee430e83c140b787c02ddd6c`:
[core](https://github.com/icip-cas/ShortGPT/blob/9a2ee8dfe9e28d0bee430e83c140b787c02ddd6c/shortgpt/shortgpt.py),
[importance metric](https://github.com/icip-cas/ShortGPT/blob/9a2ee8dfe9e28d0bee430e83c140b787c02ddd6c/shortgpt/metrics.py).
No repository license was found in this snapshot, so its source is linked rather
than redistributed. The local paper remains available.

The core imports torch, NumPy and Transformers; model device mapping needs
accelerate. The example additionally uses datasets and notebook tqdm. It downloads
PG19 and Llama-2, hard-codes CUDA, and does not provide a complete Qwen export
adapter. For an offline implementation use `/input/model`, the supplied calibration
training split and local tokenization; measure block influence before removing
layers. Rebuild layer indices, model config and attention/GDN cache mappings.
Verify reload and cached generation before scoring. Merely deleting entries in
`model.layers` is not a validated hybrid-model compression implementation.

See [tool environments](../../runtime/tools.md) and [Tool availability](../../runtime/tools.md).

## Paper reading notes

Layer redundancy and depth pruning. See [structural reduction](../README.md).
[Paper](https://arxiv.org/abs/2403.03853) · [text](paper.txt) · [PDF](paper.pdf).
Hybrid attention/state layers require architecture-specific removal and cache handling.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../references/papers.json)

A local, model-name-independent [score_layers.py](score_layers.py) now computes
mean block influence using hooks. It is our implementation of the metric, not a
copy of the unlicensed author repository. Reuse `/opt/quantize/bin/python`:

```bash
/opt/quantize/bin/python /opt/quant-docs/pruning/shortgpt/score_layers.py \
  --model /input/model --calibration /app/work/train.json \
  --layers-path model.layers --output /app/work/block-influence.json --device cuda
```

`model.layers` is an example: inspect `named_modules()` and select the actual text
decoder ModuleList. Input blocks must come from public training data, not evaluation
inputs. The report gives mean scores and ascending rank. It deliberately does not
remove layers: rebuilding caches, layer types and saved config still requires an
architecture-aware export. No separate ShortGPT environment is necessary.

[Tool availability](../../runtime/tools.md)
