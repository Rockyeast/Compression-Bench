# LLM Compressor: local API examples

LLM Compressor 0.13.0. Learn the entry points, then choose an API example as needed.

- [Schemes and algorithms](llmcompressor-local-examples.md#concepts)
- [model_free_ptq and FP8 example](llmcompressor-local-examples.md#model-free-ptq)
- [General oneshot usage](llmcompressor-local-examples.md#oneshot)
- [9B GPTQ: calibration, export and Checker](llmcompressor-local-examples.md#gptq-example)

<a id="concepts"></a>

## Schemes and algorithms

A scheme describes weight/activation precision and how scales are shared.
W8A16 means 8-bit weights and 16-bit activations; W8 alone does not specify
integer versus floating-point storage. Per-channel, per-group and per-block
scales produce different metadata and runtime requirements.

The release lists RTN-style PTQ, GPTQ, AWQ, SmoothQuant, AutoRound and rotation
methods. Choose a method and check that its dependencies and calibration
resources are available in the offline environment.

<a id="model-free-ptq"></a>

## model_free_ptq

`model_free_ptq` operates on checkpoint tensors directly for data-free schemes.
It takes a local checkpoint path as `model_stub`, an output directory as
`save_directory`, a `scheme` and optional layer exclusions in `ignore`.
It processes weights directly. Use a model-based entry point such as oneshot
for algorithms requiring calibration forward passes.

Inspect its installed signature and docstring before constructing a call:

```python
import inspect
from llmcompressor import model_free_ptq
print(inspect.signature(model_free_ptq))
print(inspect.getdoc(model_free_ptq))
```

<a id="fp8-example"></a>

### Local FP8_BLOCK example

Run with `/opt/quantize/bin/python`. Fill in an existing local small model path.
Check supported FP8 block dimensions and scheme compatibility first.
Adapt the demonstration ignore list to the selected model.

```python
from pathlib import Path
from llmcompressor import model_free_ptq

source = Path('/app/work/your-small-local-model')  # replace before running
output = Path('/app/work/smoke-fp8-model')
assert (source / 'config.json').is_file(), 'Provide a local model first'
assert not output.exists(), 'Use a fresh scratch output directory'
model_free_ptq(
    model_stub=str(source), save_directory=str(output),
    scheme='FP8_BLOCK', ignore=['lm_head'], device='cuda:0', max_workers=1,
)
assert (output / 'config.json').is_file()
assert any(output.glob('*.safetensors'))
```

This API saves the compressed checkpoint; evaluate quality and speed after reload.

This API example requires a local model compatible with the chosen scheme and
loader. After quantization exits, reload the exported checkpoint and check
generation before measuring quality and speed.
The [9B workflow](llmcompressor-local-examples.md#gptq-example) shows packaging and Checker
calls for the provided 9B resource. See the [model_free_ptq guide](llmcompressor-model-free.md).

<a id="oneshot"></a>

## General oneshot usage

`oneshot` applies a recipe to a model. A recipe supplies modifiers such as
`QuantizationModifier` or `GPTQModifier`; algorithms requiring calibration also
need representative local samples. The resulting checkpoint must include its
updated quantization configuration. The task supplies the official tokenizer
during evaluation packaging; the Agent does not need to submit tokenizer files.

In LLM Compressor 0.13.0, `oneshot` is the entry point and `recipe` selects
the algorithm modifiers to apply.

### Model loading

`model` accepts a loaded Transformers model or a model path. A path still
requires model construction/loading internally. Use provisioned local files
and a supported architecture.

### Recipe and calibration

| Input | Role |
| --- | --- |
| `recipe` | Modifier(s), such as `GPTQModifier`, and their settings |
| `dataset` | Calibration data when required by the recipe |
| `tokenizer` / `processor` | Preprocessing appropriate to the model |
| `num_calibration_samples`, `max_seq_length` | Calibration sample count and sequence length |

Calibration requirements depend on the recipe. For GPTQ, pass calibration data
through `dataset`. Data-free recipes can omit calibration data.

### Export and reload

`output_dir` requests output saving; `save_compressed=True` requests compressed
saving. The local example uses these arguments. Alternatively, after `oneshot`,
save a loaded model with `model.save_pretrained(..., save_compressed=True)`.

See the [9B GPTQ example](llmcompressor-local-examples.md#gptq-code)
and its Checker commands.
For weight-file processing without model construction, see the
[model_free_ptq guide](llmcompressor-model-free.md) and
[local FP8 example](llmcompressor-local-examples.md#fp8-example).

<a id="gptq-example"></a>

## 9B GPTQ: calibration, export and Checker

For data-free quantization, see the [FP8_BLOCK API example](llmcompressor-local-examples.md#fp8-example).
The calibrated workflow below uses the provisioned Qwen3.5-9B and WikiText train.
Use it to practice calibration, export and reload.

Use the provided 9B resource to check calibration, export and inference integration.
The public Checker reports package size, generation, PPL and decode latency using
the task's configured runtime environment and inputs. Follow the task instruction for final
evaluation of the formal model.

### Resources and output

Run inside the Agent environment with the optional 9B resource enabled:

- Original model and tokenizer: `/input/debug-model/`.
- Calibration: `/opt/calibration/train.parquet` (WikiText-2 **train**).
- Fresh debug submission: `/app/debug-submission/`; do not overwrite a candidate.
- Starter inference implementation: `/app/inference.py`.

If 9B is absent, skip this example; do not substitute the formal model or download
another model. The example uses 16 non-overlapping 512-token blocks to exercise
the workflow; adjust calibration coverage according to measured quality and cost. The
task calibration README at `/opt/calibration/README.md` also shows 128×2048 sampling.
Calibration is separate from C4 evaluation inputs. Use the tokenizer of the
model being quantized.

<a id="gptq-code"></a>

### GPTQ on the provisioned 9B model

Save this block as `/app/work/debug9b_gptq.py`. Its architecture class and
sequential targets are specific to the provisioned Qwen3.5-9B.

```python
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier


def main():
    source = Path('/input/debug-model')
    submission = Path('/app/debug-submission')
    output = submission / 'model'
    assert (source / 'config.json').is_file(), '9B resource is not enabled'
    submission.mkdir(parents=True, exist_ok=True)
    assert not any(submission.iterdir()), 'Use an empty debug submission directory'
    torch.manual_seed(42)
    torch.set_num_threads(8)

    # Calibrate on training text with the 9B tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False)
    rows = pq.read_table('/opt/calibration/train.parquet', columns=['text'])['text'].to_pylist()
    ids = tokenizer('\n\n'.join(rows), add_special_tokens=False).input_ids
    count, length = 16, 512
    starts = random.Random(42).sample(range(0, len(ids) - length + 1, length), count)
    samples = [
        {'input_ids': torch.tensor(ids[s:s + length], dtype=torch.long),
         'attention_mask': torch.ones(length, dtype=torch.long)}
        for s in starts
    ]
    dataset = DataLoader(samples, batch_size=1, shuffle=False)
    del ids, rows

    # Preserve the architecture; leave the head and special modules unquantized.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        source, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True, trust_remote_code=False)
    recipe = GPTQModifier(
        targets='Linear', scheme='W4A16', actorder=None,
        ignore=['lm_head', 're:.*visual.*', 're:.*vision.*', r're:^mtp\..*',
                r're:.*linear_attn\.in_proj_a.*', r're:.*linear_attn\.in_proj_b.*'],
        dampening_frac=0.01, block_size=128,
    )
    oneshot(
        model=model, tokenizer=tokenizer, dataset=dataset, recipe=recipe,
        num_calibration_samples=count, max_seq_length=length, batch_size=1,
        shuffle_calibration_samples=False, pipeline='sequential',
        sequential_targets=['Qwen3_5DecoderLayer'], sequential_offload_device='cpu',
        output_dir=str(output), save_compressed=True,
    )
    assert (output / 'config.json').is_file()
    assert any(output.glob('*.safetensors'))
    shutil.copyfile('/app/inference.py', submission / 'inference.py')


if __name__ == '__main__':
    main()
```

```bash
/opt/quantize/bin/python /app/work/debug9b_gptq.py
```

This exports compressed weights and matching config and copies the complete
starter interface while retaining the model's language and vision layers.
Checker uses the official 9B tokenizer. Wait for quantization to exit and release GPU memory before checking.

<a id="reload"></a>

### Reload and measure with the public Checker

After successful quantization, run inside a task with its Checker socket available:

```bash
python3 /app/check_submission.py /app/debug-submission --debug-9b --package-only
python3 /app/check_submission.py /app/debug-submission --debug-9b
```

The first command checks package and size on CPU. The second reloads the saved
submission and measures generation, PPL and decode latency through the evaluation
path. Inspect the JSON results and errors for each stage.
Use 9B results to debug the implementation; evaluate the formal 27B model separately.

For the formal model, independently choose and validate a recipe, save under
`/app/submission/`, and check without `--debug-9b`. Never submit 9B as the formal
answer. See [parameter choices](../../quantization/README.md#parameters) and [debugging](../../runtime/debugging.md).


## Explicit GPTQ weight scheme

This constructs an explicit INT4, symmetric, group-128 scheme. It replaces the
GPTQ example's `scheme='W4A16'` declaration; retain the model-appropriate `ignore`
list and calibration workflow. Verify the scheme by reloading and running inference.

```python
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor.modifiers.gptq import GPTQModifier

weight_scheme = QuantizationScheme(
    targets=['Linear'],
    weights=QuantizationArgs(num_bits=4, type='int', symmetric=True,
                             strategy='group', group_size=128),
)
recipe = GPTQModifier(config_groups={'weights': weight_scheme},
                      ignore=['lm_head'], actorder=None)
```

