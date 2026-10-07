#!/usr/bin/env python3
"""Create the 16 GiB W4A16 reference, including INT4 output-head weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_IGNORE = (
    "re:.*embed_tokens.*",
    "re:.*norm.*",
    "re:.*linear_attn\\.conv1d.*",
    "re:.*linear_attn\\.in_proj_a.*",
    "re:.*linear_attn\\.in_proj_b.*",
    "re:.*visual.*",
    "re:.*vision.*",
)


def include_output_head_target(config: dict) -> None:
    """vLLM's ParallelLMHead does not match the generic 'Linear' target."""
    quantization = config["quantization_config"]
    for group in quantization["config_groups"].values():
        targets = group.get("targets", [])
        # Qwen's multimodal wrapper names it language_model.lm_head in vLLM.
        if "Linear" in targets and "re:.*lm_head$" not in targets:
            targets.append("re:.*lm_head$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a local BF16 checkpoint to a W4A16 submission."
    )
    parser.add_argument("--model", type=Path, required=True, help="Local BF16 model directory")
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="New output directory containing model/ and inference.py",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--ignore",
        action="append",
        help="Layer name or regex kept out of INT4; repeat this option to replace defaults",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model.resolve()
    submission_dir = args.submission.resolve()
    output_model = submission_dir / "model"

    if not model_dir.is_dir():
        raise SystemExit(f"BF16 model directory does not exist: {model_dir}")
    if submission_dir.exists() and any(submission_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty submission: {submission_dir}")

    try:
        from llmcompressor import model_free_ptq
    except ImportError as exc:
        raise SystemExit(
            "llmcompressor is missing; install environment/requirements.txt first"
        ) from exc

    submission_dir.mkdir(parents=True, exist_ok=True)
    model_free_ptq(
        model_stub=str(model_dir),
        save_directory=str(output_model),
        scheme="W4A16",
        ignore=list(dict.fromkeys((args.ignore or list(DEFAULT_IGNORE)) + ["re:.*visual.*", "re:.*vision.*"])),
        max_workers=args.max_workers,
        device=args.device,
    )
    # model_free_ptq quantizes the head but exports only the 'Linear' target.
    # Explicitly select ParallelLMHead at load time; weight bytes are unchanged.
    config_path = output_model / "config.json"
    config = json.loads(config_path.read_text())
    include_output_head_target(config)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    inference_path = submission_dir / "inference.py"
    shutil.copyfile(Path(__file__).with_name("inference.py"), inference_path)
    inference_path.chmod(0o755)
    print(f"Created INT4 submission: {submission_dir}")


if __name__ == "__main__":
    main()
