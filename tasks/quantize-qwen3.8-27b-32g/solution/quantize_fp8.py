#!/usr/bin/env python3
"""Create an FP8_BLOCK reference submission from a local BF16 checkpoint."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_IGNORE = (
    "lm_head",
    "re:.*embed_tokens.*",
    "re:.*norm.*",
    "re:.*linear_attn\\.conv1d.*",
    "re:.*linear_attn\\.in_proj_a.*",
    "re:.*linear_attn\\.in_proj_b.*",
    "re:.*visual.*",
    "re:.*vision.*",
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a local BF16 checkpoint to an FP8_BLOCK submission."
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
        help="Layer name or regex kept out of FP8; repeat this option to replace defaults",
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
        scheme="FP8_BLOCK",
        ignore=list(dict.fromkeys((args.ignore or list(DEFAULT_IGNORE)) + ["re:.*visual.*", "re:.*vision.*"])),
        max_workers=args.max_workers,
        device=args.device,
    )
    inference_path = submission_dir / "inference.py"
    shutil.copyfile(Path(__file__).with_name("inference.py"), inference_path)
    inference_path.chmod(0o755)
    print(f"Created FP8 submission: {submission_dir}")


if __name__ == "__main__":
    main()
