# GPTQ with accelerated quantization

The shared entry [quantize.py](quantize.py) can be called by an Agent or a human.
It accelerates **checkpoint creation**, independently of inference kernels such
as Marlin or Machete. It uses upstream Triton GPTQ updates where supported and
`batched_quantization='auto'` for compatible same-shape matrices. Calibration
text batch defaults to **1**; matrix batching is a separate setting.

## Environment

The shared image build provisions `/opt/gptq`, isolated from `/opt/quantize` and
the base inference Python. It pins LLM Compressor commit
`b4429966e082f6e8d6f04c670c2547354b73df73`, compressed-tensors
`0.19.1a20261002`, and Transformers `5.16.1`.
See [tool environments](../../runtime/tools.md) for interpreter selection and
compatibility limits.

Outside the image, use an environment with the same pins and the repository
script path. There are no hard-coded server paths or experiment dependencies.

## Inspecting the installed implementation

Use the method's interpreter to locate its quantization API:

```bash
/opt/gptq/bin/python - <<'PYTHON'
import inspect
from llmcompressor.modifiers.gptq import GPTQModifier
print(inspect.signature(GPTQModifier))
print(inspect.getsourcefile(GPTQModifier))
PYTHON
```

## Calling the tool

Supply the local floating-point checkpoint, public training calibration data,
text projection names/patterns, and decoder block class for that architecture:

```bash
/opt/gptq/bin/python /opt/quant-docs/quantization/gptq/quantize.py \
  --model /input/model --output /app/work/gptq-candidate \
  --calibration /opt/calibration/train.parquet \
  --targets 're:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$' \
            're:.*mlp\.(gate_proj|up_proj|down_proj)$' \
  --sequential-targets LlamaDecoderLayer \
  --bits 4 --group-size 128 --dampening-frac 0.01 --batch 1
```

This illustrates Llama-style names, not a task recommendation. For another
architecture, inspect its module tree and change targets and block class.
Hybrid attention models can have additional text projections. The script prints
the resolved target list in `request.json`; large lists are factored into an
exact-equivalent regex to avoid quadratic upstream matching. Unmatched modules remain floating
point. Standard vision paths are rejected as targets, but custom modality names
still require inspection. The model loader dispatches through Transformers Auto
classes; upstream sequential tracing must also support the architecture.

Calibration accepts a `text` parquet column or JSON/NPY rows of token IDs created
with the original tokenizer. It selects `--samples` seeded blocks of `--seqlen`
tokens (defaults 128 and 2048). Use training data, not evaluation inputs.

`--bits`, `--group-size`, `--dampening-frac`, `--asymmetric`, `--ignore`, and
`--batch` are adjustable. Higher calibration batch is not required for the
accelerated kernels. Unsupported shapes/quantization modes can fall back to
upstream non-Triton computation. This wrapper exports a fixed-bit recipe, not
the experimental custom mixed-bit format. Different bit widths still require
compatible inference packing and kernels.

## Output

Choose a fresh output directory. `model/` contains the packed checkpoint;
`request.json` and `result.json` contain parameters, versions and timings.
Source model configuration is retained with exported quantization metadata;
the tool does not remove model components or compress embeddings/output head
unless explicitly targeted. It does not generate a submission inference adapter.

Reload the exported checkpoint in the target inference environment, provide the
required `inference.py`, then use the public Checker. Successful quantization
alone does not establish capacity, quality or decode performance.

## MoE and shared-projection export

Model loading uses LLM Compressor's `load_context` for the actual AutoModel class.
This exposes supported fused MoE experts as Linear modules **before** target
resolution. Inspect `resolved_targets` in `request.json`: targeting only dense
MLP names does not include `mlp.experts.<id>.*`. Choose explicit expert patterns
when intended. The upstream architecture mappings, calibration and inference
backend must all support the selected model. Sample count is a quality parameter;
reducing it is not a substitute for checking expert coverage or exported quality.

For Gemma4 `attention_k_eq_v` full-attention layers, vLLM 0.28 needs an explicit
fused `qkv_proj` target when the exporter lists exact Q/K names but there is no
separate V module. This entry adds that metadata alias when Q and K share one
quantization group. Weights and precision are unchanged; aliases are recorded in
`result.json`. A checkpoint exported elsewhere can use `add_shared_kv_targets`
from this script on a **copy** of its config. This helper only handles this
specific shared-projection layout; it does not repair arbitrary export errors.
Always reload and compare probabilities after export: a successful load can
still leave an unmatched fused projection on the wrong execution path.

Large target lists are factored during quantization only. Export restores exact module names so runtime loaders can map Hugging Face names to their own module names.
