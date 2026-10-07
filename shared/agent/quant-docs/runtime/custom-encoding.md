# Custom encoding and vLLM integration

Use the existing inference API for custom encodings. Supply the encoding
configuration, a reader and a matching computation implementation.

## Submission files

Keep config.json and SafeTensors weights under model/. SafeTensors can store
encoded integer tensors; the submission must explain their decoding semantics.
Place required Python helpers and kernel sources alongside inference.py or in
a submitted package. All retained files count toward the size limit.
Load weights using model_dir, and locate helpers relative to __file__.
The evaluation directory may be read-only. Use the writable working directory
or a temporary directory for plugin metadata and compilation caches.
Do not modify the evaluator or its installed packages.

For official operator registration and dispatch details, see the
[CustomOp document](../references/docs/vllm-custom-op.md).

## vLLM integration

For vLLM 0.28.0, register a QuantizationConfig subclass with
register_quantization_config. Its get_quant_method selects a layer implementation.
The layer's create_weights allocates the encoded parameters and attaches the
appropriate weight loaders; apply computes from those parameters.
Implement configuration parsing and the supported dtype/hardware declarations too.

A storage encoding must align fused QKV/gate-up projections,
tensor partitioning, scales and checkpoint-to-layer names.
Start with a supported unfused layer and TP=1, and reject unsupported cases explicitly.
State the tested layers, architectures and TP sizes.

Register the implementation before engine construction in both the caller and
vLLM workers. Define configuration and cross-process classes at module scope so
spawn can import and deserialize them.
One option is vLLM general plugins: include the plugin module in the submission,
arrange local discovery at load time and preserve existing plugins.
The benchmark bridge supports sibling imports and starts Python with isolated
mode -I, which ignores PYTHONPATH. Verify plugin discovery in workers separately.

## Integration checks

1. Test from a clean evaluation environment using only the submitted artifact.
2. Confirm the expected loading and compute implementation really executed in workers.
3. Compare a small case against an independent computation of the SAME decoded
   quantized weights.
4. Exercise generate, token_logprobs and stream_generate through the task runtime.
5. Check that missing registration or invalid encoding produces an explicit error.
6. Test the intended final compilation settings and hardware separately.

Loading diagnostics can be unknown or mismatched for an unrecognized encoding;
numerical correctness and approval require separate validation and review.
Follow the task instruction's model-structure constraints. The submitted model
must remain compatible with the official tokenizer, inference API, supplied
environment and package requirements.
