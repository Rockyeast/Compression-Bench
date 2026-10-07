# GPTAQ: compensate for errors from earlier quantized layers

Use `/opt/gptaq/bin/python` for **GPTQModel 7.5.0**, the integration linked by
the [GPTAQ authors](https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ).
The library implements asymmetric calibration, quantization updates and packing.
Our [entrypoint](quantize.py) connects local model/data paths and
text-only calibration to that implementation. It is an editable example, not a
required task solution or a reproduction of the paper's quality results.

## Model portability

The entry selects the registered GPTQModel 7.5.0 model definition using the local
`config.json` model_type; it no longer hard-codes Qwen. Unknown model types fail
before calibration instead of using an unreviewed automatic module tree. The
upstream definition still determines supported layers, MoE behavior and excluded
modalities. Inspect it for each new architecture: registration alone is not proof
of quantization, export or scoring-runtime compatibility. Nested text quantization
configs are rejected; start from floating-point weights. The temporary processor
flag applies only to the selected definition and is restored after failure too.
The Qwen-specific layer list below describes that definition, not every model.

## Run

```bash
cp /opt/quant-docs/quantization/gptaq/quantize.py /app/work/gptaq_quantize.py
/opt/gptaq/bin/python /app/work/gptaq_quantize.py \
  --model /input/debug-model --output /app/work/gptaq-9b \
  --method gptaq --bits 4 --group-size 128 --samples 16 --seqlen 512 \
  > /app/work/gptaq-9b.log 2>&1
```

The optional 9B resource must be provisioned. Use `/input/model` for the formal
model, with calibration and memory budgets appropriate to it. Choose a fresh
output directory. The script uses only the selected local model and public
`/opt/calibration/train.parquet` (WikiText train), with offline flags enabled.
Calibration uses seeded, non-overlapping blocks and the original tokenizer.

`--method gptaq` enables `GPTAQConfig(alpha=0.25)` by default: compare the
quantized path against floating-point reference activations. `--method gptq`
disables that compensation while keeping the remaining recipe and calibration
the same. Run the two choices into separate directories for a controlled test.
`--alpha` adjusts the author's compensation coefficient; it is not an error rate.

The CLI exposes the library's integer options 2/3/4/8 and group sizes 32/64/128.
Start with INT4/group128; other combinations require their own export/reload and
quality checks in the target runtime. Activations remain BF16. The upstream
Qwen3.5 layer definition selects the text decoder projections; vision, output
head, normalization, recurrent convolution and small `in_proj_a/b` controls remain
unquantized. Fixed bitwidth avoids incompatible choices within fused projections.
The small sample budget above is for trying the workflow, not a quality recipe.

## Output and inference

- `request.json`: input paths, recipe and version.
- `quantization-log.json`: upstream per-layer quantization diagnostics.
- `model/`: packed checkpoint exported by GPTQModel.
- `result.json`: export path and elapsed time, written after a successful save.

The **algorithm** is GPTAQ; the saved **format** is `FORMAT.GPTQ`. `GPTQ_V2` is a
separate checkpoint-layout option and is not the switch for the GPTAQ algorithm.
This pinned version uses `gptaq=GPTAQConfig()`, rather than older examples' `v2=True`.

Place the exported checkpoint in the submission's `model/` with a compatible
`inference.py`. Reload it using the inference environment (`python3`), then run
the [public Checker](../../runtime/inference-api.md). Keep work logs outside the submission.
Use the same model and precision for generation, probabilities and decoding.

## Environment and scope

GPTQModel lives in its own venv. Installation keeps the base Torch, Transformers,
vLLM, compressed-tensors and Humming versions fixed; Compressor and AutoRound
environments are separate. No Checker, Verifier or scoring changes are required.

The wrapper temporarily skips image-processor construction for pretokenized
text-only calibration, since the provided checkpoint need not include vision
processor assets. It retains the original model structure and upstream layer
walker and restores the flag afterwards. It does not implement tokenizer fixes
or modify installed library source. Model coverage follows registered upstream definitions;
new architectures still require export, reload and quality validation.

GPTAQ keeps floating-point reference data as well as the quantized path. Measure
its memory/time requirements before allocating a large calibration budget.
PPL, answer quality and decode speed must be measured on the exported candidate.

## Paper reading notes

Yuhang Li, Ruokai Yin, Donghyun Lee, Shiting Xiao and Priyadarshini Panda.
*GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration*
(ICML 2025).
[Paper](https://proceedings.mlr.press/v267/li25dn.html) ·
[Implementation](https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ).

Use the isolated GPTQModel environment and the offline [GPTAQ entrypoint](README.md)
to try the author-linked implementation on local resources.

- **Idea:** a quantized layer receives inputs already perturbed by earlier
  quantized layers. Match its output to the full-precision model's output to
  compensate for both local and accumulated errors.
- **Read:** asymmetric calibration and section 4, GPTAQ Methodology, including
  the weight update and efficient implementation.
- **Possible experiment:** hold bitwidths/group sizes fixed and compare ordinary
  GPTQ with accumulated-error compensation.
- **Implementation work:** paired full-precision/quantized calibration activations
  and changes to the quantization update. Inference can reuse an existing backend
  if the exported tensors/config follow a format that backend supports.
  The algorithm's earlier name is GPTQv2. The similarly named `gptq_v2`
  refers to a GPTQ checkpoint format.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../references/papers.json)

Dedicated entry: [quantize.py](quantize.py).

[Tool availability](../../runtime/tools.md)
