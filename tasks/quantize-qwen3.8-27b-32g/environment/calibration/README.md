# Optional calibration data

`train.parquet` contains the unmodified `train` split of
`Salesforce/wikitext`, subset `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.

Source: https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet

Original text: Wikipedia contributors; distributed via WikiText by Salesforce.
License: CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/).

Use this optional resource for data-dependent quantization. Calibration data and
evaluation inputs are separate. Keep calibration files outside the submission;
evaluated inference must use only submitted model assets and the preinstalled runtime.

The example samples 128 non-overlapping 2048-token blocks with seed 42.
Adjust sample count and length to suit your method and compute budget.
Run with `/opt/quantize/bin/python`:

```python
import random
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# For optional 9B calibration, use "/input/debug-model" instead.
tokenizer = AutoTokenizer.from_pretrained("/input/model", local_files_only=True)
rows = pq.read_table("/opt/calibration/train.parquet", columns=["text"])["text"].to_pylist()
ids = tokenizer("\n\n".join(rows), add_special_tokens=False).input_ids
starts = random.Random(42).sample(range(0, len(ids) - 2048 + 1, 2048), 128)
samples = [
    {"input_ids": ids[s:s + 2048], "attention_mask": [1] * 2048}
    for s in starts
]
```
