"""Offline AutoScheme allocation + AutoRound tuning/export for locally supported Transformers models.

Run with /opt/autoround/bin/python. See README.md for budget semantics.
"""
import argparse
from contextlib import contextmanager
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re

SUPPORTED_BITS = (2, 3, 4, 8)


def require_export_versions():
    if importlib.metadata.version("auto-round") != "0.14.2" or importlib.metadata.version("compressed-tensors") != "0.18.0":
        raise RuntimeError("Export adapter requires auto-round 0.14.2 and compressed-tensors 0.18.0 (dense INT3 packing)")


def validate_integer_layers(layer_config):
    """Keep the local export adapter deliberately narrower than arbitrary AutoRound schemes."""
    quantized = 0
    for name, layer in layer_config.items():
        bits = layer.get("bits", 16)
        if bits == 16:
            continue
        if (bits not in SUPPORTED_BITS or layer.get("data_type", "int") != "int"
                or layer.get("act_bits", 16) != 16 or layer.get("sym") is not True
                or not isinstance(layer.get("group_size"), int) or layer["group_size"] <= 0
                or layer.get("super_bits") is not None):
            raise ValueError(f"Unsupported integer-export configuration: {name}: {layer}")
        quantized += 1
    if not quantized:
        raise ValueError("Allocation contains no quantized matrices; reduce the budget")


@contextmanager
def integer_export_adapter(layer_config):
    """Reuse upstream packing/saving with a task-local INT2/3/4/8 format adapter.

    No installed library files or inference kernels are modified. The upstream
    CT format gate only admits 4/8; its generic packer takes each matrix's bits.
    Restore registration even on export errors. Only use in this single-run CLI.
    """
    validate_integer_layers(layer_config)
    require_export_versions()
    from auto_round.formats import LLMCompressorFormat, OutputFormat

    class IntegerFormat(LLMCompressorFormat):
        def __init__(self, format, ar):
            validate_integer_layers(ar.layer_config)
            self.output_format = "llm_compressor:wint_a16"
            self.backend = None

    previous = OutputFormat._format_list["llm_compressor"]
    OutputFormat.register("llm_compressor")(IntegerFormat)
    try:
        yield
    finally:
        OutputFormat._format_list["llm_compressor"] = previous


def shared_groups(names):
    """Keep projections fused by the starter backend at identical precision."""
    names = set(names)
    groups = []
    for name in sorted(names):
        parent, _, leaf = name.rpartition(".")
        for first, siblings in (
            ("q_proj", ("q_proj", "k_proj", "v_proj")),
            ("gate_proj", ("gate_proj", "up_proj")),
            ("in_proj_qkv", ("in_proj_qkv", "in_proj_z")),
        ):
            if leaf == first:
                group = [parent + "." + item for item in siblings]
                if all(item in names for item in group):
                    groups.append(group)
    return groups


def load_float_model(path):
    """Dispatch through Transformers; preserve the entire multimodal checkpoint."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if getattr(config, "quantization_config", None) or getattr(
            getattr(config, "text_config", None), "quantization_config", None):
        raise ValueError("Start from unquantized floating-point weights")
    loader = (AutoModelForImageTextToText if getattr(config, "vision_config", None) is not None
              else AutoModelForCausalLM)
    if type(config) not in loader._model_mapping:
        raise ValueError(f"No {loader.__name__} mapping for {config.model_type!r}; provide a model adapter")
    return loader.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                  local_files_only=True, trust_remote_code=False)


def layer_policy(names, extra_excludes=(), group_override=None):
    """Conservative defaults plus explicit backend-specific fusion constraints."""
    names = set(names)
    patterns = [re.compile(p) for p in extra_excludes]
    fixed = {name: {"bits": 16, "act_bits": 16} for name in names
             if any(marker in name for marker in ("visual", "vision", "audio", "multimodal",
                                                   "mtp", "lm_head", "in_proj_a", "in_proj_b"))
             or any(p.search(name) for p in patterns)}
    eligible = names - fixed.keys()
    groups = shared_groups(eligible) if group_override is None else group_override
    if not isinstance(groups, list):
        raise ValueError("Shared layers must be a JSON list of groups")
    used = set()
    for group in groups:
        if (not isinstance(group, list) or len(group) < 2
                or any(not isinstance(n, str) for n in group)):
            raise ValueError("Each shared group needs at least two module names")
        if len(set(group)) != len(group) or not set(group) <= eligible or used.intersection(group):
            raise ValueError("Shared groups must be disjoint, present, non-excluded linear modules")
        used.update(group)
    if not eligible:
        raise ValueError("No eligible linear layers remain")
    return fixed, groups


def sample_blocks(ids, count, length, seed):
    starts = range(0, len(ids) - length + 1, length)
    if len(starts) < count:
        raise ValueError(f"Calibration contains {len(starts)} full blocks; requested {count}")
    return [ids[s:s + length] for s in random.Random(seed).sample(starts, count)]


def json_value(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def explicit_export_targets(config, layer_config):
    """Bind each exported precision to its actual matrices, not overlapping 'Linear' targets.

    AutoRound 0.14.2's CT exporter can emit multiple groups targeting all Linear
    modules. vLLM then selects one group's bit width for differently packed weights.
    Repair metadata only, using the allocator's actual final configuration.
    """
    groups = config["quantization_config"]["config_groups"]
    targets = {key: [] for key in groups}
    for name, layer in layer_config.items():
        bits = layer.get("bits", 16)
        if bits >= 16:
            continue
        if layer.get("data_type", "int") != "int" or layer.get("act_bits", 16) != 16:
            raise ValueError(f"Unexpected non-integer/activation scheme for {name}")
        matches = []
        for key, group in groups.items():
            weights = group["weights"]
            if (weights["num_bits"] == bits and weights["type"] == "int"
                    and weights["group_size"] == layer["group_size"]
                    and weights["symmetric"] == layer["sym"]
                    and group.get("input_activations") is None):
                matches.append(key)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one export group for {name}, got {matches}")
        targets[matches[0]].append(name)
    if any(not names for names in targets.values()):
        raise ValueError("An exported group has no matching quantized matrices")
    for key, names in targets.items():
        groups[key]["targets"] = sorted(names)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Fresh work directory, outside submission")
    parser.add_argument("--calibration", type=Path, default=Path("/opt/calibration/train.parquet"))
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8], choices=SUPPORTED_BITS,
                        help="Available integer weight widths; excluded matrices remain BF16")
    parser.add_argument("--avg-bits", type=float, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--iters", type=int, default=20, help="Rounding-tuning steps; 0 exercises allocation + RTN export")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", action="append", default=[], metavar="REGEX",
                        help="Additional linear layers kept at BF16; repeatable")
    parser.add_argument("--shared-layers", type=Path,
                        help="JSON list of same-precision groups; replaces sibling-name defaults")
    args = parser.parse_args()
    if not (args.model / "config.json").is_file() or not args.calibration.is_file():
        parser.error("Provide an existing local model and calibration parquet")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh work directory")
    if min(args.samples, args.seqlen, args.group_size) <= 0 or args.iters < 0:
        parser.error("samples, seqlen and group-size must be positive; iters must be nonnegative")
    if len(set(args.bits)) < 2 or not 0 < args.avg_bits <= 16:
        parser.error("Choose at least two bit options and an average budget in (0, 16]")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    require_export_versions()

    import pyarrow.parquet as pq
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from auto_round import AutoRound, AutoScheme, QuantizationScheme
    from auto_round.calib_dataset import register_dataset

    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    rows = pq.read_table(args.calibration, columns=["text"])["text"].to_pylist()
    ids = tokenizer("\n\n".join(row for row in rows if row), add_special_tokens=False).input_ids
    blocks = sample_blocks(ids, args.samples, args.seqlen, args.seed)
    del ids, rows

    @register_dataset("quantbench_local_train")
    def local_train(tokenizer, seqlen, **kwargs):
        if seqlen != args.seqlen:
            raise ValueError("Allocation and tuning must use the same calibration length")
        return Dataset.from_dict({"input_ids": blocks, "attention_mask": [[1] * seqlen for _ in blocks]})

    args.output.mkdir(parents=True)
    # The provisioned model is text-evaluated and need not contain image-processor assets.
    # Use the author's basic template with our pretokenized text dataset; no image inputs.
    text_template = args.output / "text-template.json"
    text_template.write_text(json.dumps({"model_type": "quantbench_text_only", "processor": "basic",
                                         "default_dataset": "quantbench_local_train"}) + "\n")
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest.update(auto_round_version="0.14.2")
    (args.output / "request.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = load_float_model(args.model)
    names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    override = json.loads(args.shared_layers.read_text()) if args.shared_layers else None
    fixed, groups = layer_policy(names, args.exclude, override)
    manifest.update(model_type=model.config.model_type, excluded_layers=sorted(fixed), shared_layers=groups)
    (args.output / "request.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "shared_layers.json").write_text(json.dumps(groups, indent=2) + "\n")
    scheme = AutoScheme(
        avg_bits=args.avg_bits,
        options=[QuantizationScheme(bits=b, group_size=args.group_size, sym=True, act_bits=16)
                 for b in sorted(set(args.bits))],
        shared_layers=groups, method="DeltaLoss", nsamples=args.samples, seqlen=args.seqlen,
        batch_size=1, dataset="quantbench_local_train", low_gpu_mem_usage=True)
    compressor = AutoRound(
        model, tokenizer=tokenizer, template=str(text_template), scheme=scheme, layer_config=fixed,
        dataset="quantbench_local_train", nsamples=args.samples, seqlen=args.seqlen,
        batch_size=1, iters=args.iters, seed=args.seed, device_map=0,
        low_gpu_mem_usage=True, enable_torch_compile=False)
    # Allocation requires autograd. Do not wrap this call in inference_mode/no_grad.
    # Persist allocation before tuning/export so later failures do not lose the decision table.
    compressor.post_init()
    (args.output / "allocation.json").write_text(
        json.dumps(compressor.layer_config, indent=2, default=json_value) + "\n")
    with integer_export_adapter(compressor.layer_config):
        _, folders = compressor.quantize_and_save(str(args.output / "export"), format="llm_compressor")
    export_dirs = [folders] if isinstance(folders, str) else folders
    if len(export_dirs) != 1:
        raise ValueError(f"Expected one export directory, got {folders}")
    config_path = Path(export_dirs[0]) / "config.json"
    original_config = config_path.read_text()
    (args.output / "export-config-before-target-fix.json").write_text(original_config)
    config = explicit_export_targets(json.loads(original_config), compressor.layer_config)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "layer_config.json").write_text(
        json.dumps(compressor.layer_config, indent=2, default=json_value) + "\n")
    manifest["export_folders"] = folders
    manifest["status"] = "exported; reload and Checker required"
    (args.output / "result.json").write_text(json.dumps(manifest, indent=2, default=json_value) + "\n")
    print(json.dumps({"export_folders": folders, "layer_config": str(args.output / "layer_config.json")}, default=json_value))


if __name__ == "__main__":
    main()
