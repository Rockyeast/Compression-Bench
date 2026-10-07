# AutoRound: sensitivity-guided mixed precision

Use `/opt/autoround/bin/python` for **auto-round 0.14.2**. This separate environment
shares the base Torch runtime; LLM Compressor stays in `/opt/quantize`, and final
inference uses `python3`. See [the paper](README.md) and the
[author implementation](https://github.com/intel/auto-round).

The runnable [example](quantize.py) uses `AutoScheme(method="DeltaLoss")`:
calibration forward/backward passes estimate sensitivity, the allocator chooses
bits under a budget, and AutoRound tunes rounding before compressed export.
An allocation unit is typically a Linear weight matrix. Projections fused by
the inference backend share their bit choice. This is a starting recipe you can
edit, rather than a required task solution or a reproduction of paper results.

The example is a task-owned integration script, not a verbatim upstream example.
The AutoRound package is installed from the author's PyPI release; sensitivity,
allocation and tuning remain upstream implementations. Local code handles input
paths, calibration, fusion constraints and export compatibility.

## Model portability

The entry uses Transformers AutoConfig and AutoModel dispatch for local causal-LM
or image-text models rather than a Qwen-specific class. Unknown/custom-code models
require an explicit adapter; no remote code or models are fetched. AutoRound must
also support the selected architecture. Loading alone does not establish that.

Default precision groups recognize complete sibling q/k/v, gate/up and GDN
qkv/z projections. They are naming-based starting constraints, not universal
backend fusion rules. For another backend or naming scheme, pass `--shared-layers
/path/groups.json`, containing e.g. `[["decoder.proj_a", "decoder.proj_b"]]`.
An empty list disables the default groups; only do this if the deployment backend
supports independent precisions. Invalid, overlapping or excluded names fail.
Use repeatable `--exclude REGEX` to keep additional linear modules at BF16.
Vision/audio, output-head and known control-module exclusions remain conservative
defaults; inspect module names on every new multimodal architecture and add any
missing non-text exclusions. Non-text weights/config must remain unchanged.
`request.json` records the model type, exclusions and resolved groups for review.

## Run with local resources

Choose a calibration/tuning budget separately from the bit-width and archive-size
budget. The same script supports both profiles:

| Purpose | Model | Steps per block (`--iters`) | Samples | Tokens per sample |
| --- | --- | ---: | ---: | ---: |
| Small-scale debugging | Optional 9B | 20 | 16 | 512 |
| Formal-model quality experiment | Task model | 200 | 128 | 2048 |

The small profile checks quantization, export and reloading; it is not a quality
recommendation for the final submission. The larger profile is a starting point
for quality experiments, not a guaranteed passing recipe or an eight-hour runtime
guarantee. Both profiles use batch size 1 in this script.

### Small-scale debugging

When the optional 9B resource is present, start with a fresh work directory:

```bash
cp /opt/quant-docs/quantization/autoround/quantize.py /app/work/autoround_mixed.py
/opt/autoround/bin/python /app/work/autoround_mixed.py \
  --model /input/debug-model --output /app/work/autoround-9b \
  --bits 4 8 --avg-bits 6 --samples 16 --seqlen 512 --iters 20 \
  > /app/work/autoround-9b.log 2>&1
```

### Formal-model quality experiment

After checking the export/loading path, use the task model and a fresh output
directory. First set the shell variable `AVG_BITS` to a feasible budget for your
task, for example `AVG_BITS=6` only when the capacity permits it. There is no
single average-bit setting that fits all four task capacities. The command below
requires an explicit value instead of silently reusing the 9B example's budget.

```bash
cp /opt/quant-docs/quantization/autoround/quantize.py /app/work/autoround_mixed.py
/opt/autoround/bin/python /app/work/autoround_mixed.py \
  --model /input/model --output /app/work/autoround-formal \
  --bits 2 3 4 8 --avg-bits "${AVG_BITS:?Set AVG_BITS for this task first}" \
  --samples 128 --seqlen 2048 --iters 200 \
  > /app/work/autoround-formal.log 2>&1
```

Keep only the candidate bit widths relevant to your experiment; more candidates
also increase sensitivity-estimation work. Validate the chosen widths and grouping
with the intended inference backend before a long run. Increasing calibration or
tuning does not fix an unsupported export format.

The larger profile borrows the paper's basic tuning sample count, sequence length
and iteration count; it does not reproduce the complete paper recipe. The paper
uses batch size 8 and separate DeltaLoss calibration (16 samples of 256 tokens).
This wrapper uses batch size 1 and shares `--samples`/`--seqlen` between allocation
and tuning, so increasing these options also increases allocation cost. See the
[paper, Section 3.5](paper.pdf) for the original settings.

With an eight-hour task budget, measure allocation and representative block-tuning
times before committing the remaining time. Reserve time for export, reloading,
Checker and fixes. Adjust `--iters`, `--samples` and `--seqlen` based on measured
quality and remaining time; 200 steps are not mandatory. Changing these arguments
starts a new run, not a resume. Judge the final artifact with Checker, not the
tuning loss alone. Neither profile supplies a measured full-run time for the task
model.

### Shared data and bit-width settings

The script reads only the selected local model and `/opt/calibration/train.parquet`
(WikiText train). Both allocation and tuning use the same seeded, non-overlapping
token blocks. It sets the Hugging Face offline flags before importing libraries.
Use `--model /input/model` for the formal checkpoint after checking resource costs.

`--bits` sets available weight precisions; activations remain BF16. `--avg-bits`
is the library's parameter-weighted bit budget, including fixed-precision layers
and scale overhead. It is not a submission-size guarantee. The library prints the
achievable range; choose a target within it. Final archive size comes from Checker.
Choose any set of at least two widths from **2, 3, 4, 8**, for example
`--bits 2 3 --avg-bits 4.3` on the supplied 9B model, or `--bits 2 3 4 8`
with a budget within the printed feasible range. Excluded matrices stay BF16;
16 is not an integer option. Different models/exclusions change the feasible range.

INT2/INT3 use the script's scoped integer-export adapter for AutoRound 0.14.2 and
compressed-tensors 0.18.0. It reuses upstream packing/saving and explicitly admits
only symmetric, positive-group-size, weight-only INT2/3/4/8. This is a local
compatibility extension, not an upstream claim that every export backend supports
all widths. Installed library files and inference kernels remain unchanged.
The process-local format registration is restored after export, including errors.
The isolated AutoRound environment uses compressed-tensors 0.18.0 for dense INT3
packing; the base inference environment is unchanged. Version 0.17's padded INT3
packing differs from the current vLLM loader's expected layout.

`--iters 0` still performs sensitivity-based allocation, then exports rounded
weights without the iterative tuning stage. Positive values enable tuning;
the example's small calibration and iteration budgets are for trying the workflow.
Choose larger budgets based on measured quality and available time.

## Outputs and inference

- `request.json`: version and settings.
- `shared_layers.json`: matrix groups constrained to the same precision.
- `allocation.json`: selected matrix configurations, saved before tuning/export.
- `layer_config.json`: each matrix's final quantization configuration.
- `result.json`: actual exported folder(s); export may add a nested directory.
- The log includes allocation diagnostics and tuning progress. DeltaLoss is a
  sensitivity estimate, while Checker measures actual PPL and inference time.

Copy the actual exported checkpoint into a fresh submission's `model/`, together
with a compatible `inference.py`. Use the starter inference runtime to reload it
and then follow the [9B Checker workflow](../README.md#reload), or
the formal task Checker for the formal model. Recheck loading when changing bit
options, grouping or exclusions; a valid allocation alone does not establish
backend compatibility. Keep diagnostic JSON and logs in the work directory.

For the supplied Qwen structure, the default exclusions keep the vision tower,
MTP, output head and small recurrent-control projections in BF16. On other models,
review the selected groups and extend exclusions to preserve non-text modules. The exported format is `llm_compressor`
(compressed-tensors); no changes to the task's Checker or scoring are needed.

For mixed precision, the example writes explicit matrix names into each exported
precision group's `targets`. AutoRound 0.14.2 can otherwise emit overlapping
`Linear` targets for INT4 and INT8. This metadata normalization uses the actual
final layer configuration and leaves the packed weights unchanged.

## Paper reading notes

Wenhua Cheng, Weiwei Zhang, Heng Guo, Haihao Shen and Zaner Ma.
*SignRoundV2: Toward Closing the Performance Gap in Extremely Low-Bit
Post-Training Quantization for LLMs* (2026 revision).
[Paper](https://arxiv.org/abs/2512.04746v2) ·
[Implementation: Intel AutoRound](https://github.com/intel/auto-round).

The installed AutoRound environment and an offline AutoScheme example are
described in [autoround.md](README.md).

- **Idea:** estimate which layers are most harmed by quantization, allocate bits
  under a global budget, then optimize rounding and quantization parameters.
- **Read:** section 3, especially DeltaLoss sensitivity, layer-wise bit allocation,
  pre-tuning scale search and loss filtering; section 4.6 for quantization cost.
- **Possible experiment:** compare uniform allocation with sensitivity-guided
  allocation at comparable saved size, then separately test parameter tuning.
- **Implementation work:** calibration forwards/backwards, a budget allocator and
  block-wise optimization. Full tuning is an option under the task budget;
  measure tuning cost on the task model and GPU. Match the exported format
  to a supported loader, or include a compatible loader in the submission.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../references/papers.json)

Dedicated entry: [quantize.py](quantize.py).

[Tool availability](../../runtime/tools.md)
