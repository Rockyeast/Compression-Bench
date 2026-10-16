"""Build-time dependency check; GPU model compatibility requires separate tests."""
import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--record", action="store_true")
args = parser.parse_args()
expected = {"vllm": "0.28.0", "humming-kernels": "0.1.13", "math-verify": "0.9.0"}
packages = [*expected, "torch", "transformers", "triton", "gguf", "huggingface_hub", "vllm-gguf-plugin"]
versions = {name: version(name) for name in packages}
for name, pinned in expected.items():
    assert versions[name] == pinned, (name, versions[name], pinned)
for name in ["vllm", "vllm_gguf_plugin", "gguf", "math_verify"]:
    importlib.import_module(name)
plugin = Path(importlib.import_module("vllm_gguf_plugin").__file__).parent
assert hashlib.sha256((plugin / "ops.py").read_bytes()).hexdigest() == "d3a70e3b6ab2e3154999c89dd635aafb041510cd84f6825120dc1d3ac2716265"
mapping = (plugin / "weights_adapter/qwen3_5.py").read_text()
assert '"qwen3_5_text": "Qwen3_5ForCausalLM"' in mapping
assert '"qwen3_5_moe_text": "Qwen3_5MoeForCausalLM"' in mapping
manifest = Path(__file__).with_name("versions.json")
if args.record:
    manifest.write_text(json.dumps(versions, indent=2) + "\n")
else:
    assert versions == json.loads(manifest.read_text()), "Inference dependencies changed after base build"
print(json.dumps(versions, sort_keys=True))
