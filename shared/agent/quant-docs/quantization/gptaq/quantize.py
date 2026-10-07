"""Offline model-dispatched weight quantization via upstream GPTQModel's GPTAQ implementation.

Run with /opt/gptaq/bin/python; see README.md. No evaluation inputs are used.
"""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time


def sample_blocks(ids, count, length, seed):
    starts = range(0, len(ids) - length + 1, length)
    if len(starts) < count:
        raise ValueError(f"Only {len(starts)} calibration blocks; requested {count}")
    return [ids[start:start + length] for start in random.Random(seed).sample(starts, count)]


@contextmanager
def text_only_calibration(model_definition):
    """Task resources contain tokenizer/weights, not vision processor assets.

    Retain the full model and upstream layer walker. Only skip image-processor
    construction for our pretokenized text-only calibration; restore afterwards.
    """
    owned = "require_load_processor" in vars(model_definition)
    previous = getattr(model_definition, "require_load_processor", False)
    model_definition.require_load_processor = False
    try:
        yield
    finally:
        if owned:
            model_definition.require_load_processor = previous
        else:
            del model_definition.require_load_processor


def resolve_definition(config, model_map):
    """Use the pinned library's registered walker, never guess a module tree."""
    model_type = config.get("model_type", "").lower()
    if model_type not in model_map:
        raise ValueError(f"GPTQModel has no registered definition for {model_type!r}; "
                         "add a model adapter before quantizing this architecture")
    return model_map[model_type]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh work directory, outside submission")
    parser.add_argument("--calibration", type=Path, default=Path("/opt/calibration/train.parquet"))
    parser.add_argument("--method", choices=("gptaq", "gptq"), default="gptaq")
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 8), default=4)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--alpha", type=float, default=0.25, help="Upstream GPTAQ compensation coefficient")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(parser, args):
    if not (args.model / "config.json").is_file() or not args.calibration.is_file():
        parser.error("Provide an existing local model and calibration parquet")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    if args.samples <= 0 or args.seqlen < 16 or not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error("samples must be positive, seqlen >= 16 and alpha finite/nonnegative")
    config = json.loads((args.model / "config.json").read_text())
    if not config.get("model_type"):
        parser.error("config.json must declare model_type")
    if config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config"):
        parser.error("This example requires unquantized floating-point weights")


def main():
    parser = make_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    args.model, args.output, args.calibration = (p.resolve() for p in (args.model, args.output, args.calibration))
    if importlib.metadata.version("gptqmodel") != "7.5.0":
        raise RuntimeError("This example requires GPTQModel 7.5.0")
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from gptqmodel import GPTAQConfig, GPTQModel, QuantizeConfig, BACKEND
    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.quantization import FORMAT

    model_config = json.loads((args.model / "config.json").read_text())
    definition = resolve_definition(model_config, MODEL_MAP)
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    rows = pq.read_table(args.calibration, columns=["text"])["text"].to_pylist()
    ids = tokenizer("\n\n".join(row for row in rows if row), add_special_tokens=False).input_ids
    blocks = sample_blocks(ids, args.samples, args.seqlen, args.seed)
    del ids, rows
    calibration = [{"input_ids": block, "attention_mask": [1] * len(block)} for block in blocks]
    args.output.mkdir(parents=True)
    # Upstream writes relative diagnostic logs. Keep those in the writable work
    # directory rather than the read-only container /app or the original model.
    os.chdir(args.output)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest.update(model_type=model_config["model_type"], model_definition=definition.__name__,
                    gptqmodel_version="7.5.0",
        scope="weight-only, fixed bit width, original tokenizer, public calibration only")
    (args.output / "request.json").write_text(json.dumps(manifest, indent=2) + "\n")
    quant_config = QuantizeConfig(
        bits=args.bits, group_size=args.group_size, format=FORMAT.GPTQ,
        desc_act=False, act_group_aware=False, sym=True, lm_head=False,
        gptaq=GPTAQConfig(alpha=args.alpha, device="cuda:0") if args.method == "gptaq" else None,
        device="cuda:0", offload_to_disk=False, fallback=None,
        auto_forward_data_parallel=False, calibration_data_device="cpu")
    with text_only_calibration(definition):
        model = GPTQModel.load(str(args.model), quant_config, trust_remote_code=False,
                               local_files_only=True, dtype=torch.bfloat16)
        quant_log = model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH,
                                   calibration_sort=None, calibration_data_min_length=16)
        (args.output / "quantization-log.json").write_text(json.dumps(quant_log, indent=2, default=str) + "\n")
        model.save(str(args.output / "model"))
    saved = json.loads((args.output / "model/config.json").read_text())
    if not saved.get("quantization_config"):
        raise RuntimeError("Export has no quantization configuration")
    manifest.update(status="exported; reload and Checker required", elapsed_seconds=time.monotonic()-started,
                    export_folder=str(args.output / "model"))
    (args.output / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
