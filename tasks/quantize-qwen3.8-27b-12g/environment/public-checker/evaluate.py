#!/usr/bin/env python3
"""Evaluate submitted inference; independent review required."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr
import decode_benchmark as decode
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from inference_runtime import SubmittedRuntime, InferenceError, generate_smoke, review_metadata
from tokenizer_assets import packaged_inventory, prepared_submission
from decode_benchmark import benchmark as benchmark_decode, final_score
from ppl_benchmark import measure as measure_ppl, SCORING_PROTOCOL


GIB = 1024**3
DEFAULT_LIMIT_GIB = 12.0
TOKENIZER_DIR = Path(__file__).parent / "evaluator-assets" / "tokenizer"
SCORING_VERSION = SCORING_PROTOCOL


class SubmissionError(RuntimeError):
    pass


def inspect_submission(submission: Path, limit_gib: float, *, tokenizer_dir=None) -> dict[str, object]:
    if submission.is_symlink():
        raise SubmissionError("submission directory itself cannot be a symbolic link")
    if not submission.is_dir():
        raise SubmissionError(f"submission directory does not exist: {submission}")
    model_dir = submission / "model"
    inference = submission / "inference.py"
    if not model_dir.is_dir():
        raise SubmissionError("missing submission/model/")
    if not inference.is_file():
        raise SubmissionError("missing submission/inference.py")
    if not (model_dir / "config.json").is_file():
        raise SubmissionError("missing submission/model/config.json")
    if not any(model_dir.glob("*.safetensors")):
        raise SubmissionError("submission/model/ contains no safetensors weights")

    try:
        inventory = packaged_inventory(submission, tokenizer_dir or TOKENIZER_DIR)
    except ValueError as exc:
        raise SubmissionError(str(exc)) from exc
    total_bytes = inventory["total_bytes"]

    limit_bytes = int(limit_gib * GIB)
    if total_bytes > limit_bytes:
        raise SubmissionError(
            f"submission is too large: {total_bytes} bytes > {limit_bytes} bytes"
        )
    return {
        **inventory,
        "valid": True,
        "total_bytes": total_bytes,
        "total_gib": total_bytes / GIB,
        "limit_gib": limit_gib,
    }


def offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["VLLM_NO_USAGE_STATS"] = "1"
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return environment


@contextmanager
def evaluation_package(submission, tokenizer):
    """Prepare the official-tokenizer view; clean up on exit."""
    with prepared_submission(submission, tokenizer) as package:
        yield package


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    inspect_submission(args.submission, args.limit_gib, tokenizer_dir=args.tokenizer)
    with evaluation_package(args.submission, args.tokenizer) as package:
        output = generate_smoke(package, args.tokenizer, args.prompt, args.max_new_tokens, args.timeout)
    return {"valid": True, "generated_text": output, **review_metadata()}


def evaluate_ppl(args: argparse.Namespace) -> dict[str, object]:
    inspect_submission(args.submission, args.limit_gib, tokenizer_dir=args.tokenizer)
    with evaluation_package(args.submission, args.tokenizer) as package:
        return measure_quality(package, args.tokenizer, args.ppl_assets,
                               timeout=args.inference_timeout, split="hidden")


def evaluate_decode(args):
    inspect_submission(args.submission, args.limit_gib, tokenizer_dir=args.tokenizer)
    quality = json.loads(args.ppl_result.read_text())
    if quality.get('valid') is not True:
        raise SubmissionError('decode requires a valid quality result')
    with evaluation_package(args.submission, args.tokenizer) as package:
        return benchmark_decode(package, args.tokenizer, timeout=args.timeout,
                                samples_output=args.samples_output)


def add_ppl_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--ppl-assets", type=Path, default=Path(__file__).parent / 'evaluator-assets')
    parser.add_argument("--inference-timeout", type=float, default=600)


def measure_quality(submission, tokenizer, ppl_assets, timeout=600, split="hidden"):
    """Shared PPL measurement and reporting for both individual and full runs."""
    os.environ.update(offline_environment())
    result = measure_ppl(submission, ppl_assets, tokenizer, SubmittedRuntime,
                         timeout=timeout, expected_split=split)
    if not math.isfinite(result["ppl"]) or result["ppl"] < 1:
        raise SubmissionError("PPL must be finite and >= 1")
    if split == "hidden":
        cfg = json.loads((ppl_assets / "ppl_config.json").read_text())
        result.update(quality_reward=None, scoring_version=SCORING_PROTOCOL,
                      final_reward_source="score.json", quality_reference_ppl=cfg.get("reference_ppl"),
                      bf16_ppl=cfg.get("bf16_ppl"), baseline_status=cfg["baseline_status"],
                      runtime="submitted-inference-audited-v1",
                      reference_validation=cfg["baseline_status"])
    return {**result, **review_metadata()}


def measure_decode(package, profile, timeout, samples_output=None):
    """Select this profile's decode inputs; restore the previous selection on failure too."""
    previous_assets = decode.ASSETS_DIR
    try:
        decode.ASSETS_DIR = profile["decode_assets"]
        return benchmark_decode(package, profile["tokenizer"], timeout=timeout,
                                samples_output=samples_output)
    finally:
        decode.ASSETS_DIR = previous_assets


def score_measurements(quality, speed, profile, public):
    """Use the same formula, with the selected public/hidden protocol and input checks."""
    previous_assets = decode.ASSETS_DIR
    try:
        decode.ASSETS_DIR = profile["decode_assets"]
        return final_score(quality, speed, public=public, ppl_assets=profile["ppl_assets"])
    finally:
        decode.ASSETS_DIR = previous_assets


def evaluate_submission(submission, profile, *, public=False, package_only=False,
                        timeout=600, output_dir=None):
    """Shared check -> generation -> PPL -> decode -> score; only hidden runs save stage logs."""
    results = {}

    def stage(name, operation):
        log_context = ((output_dir / f"{name}.log").open("w", encoding="utf-8")
                       if output_dir is not None else nullcontext(None))
        with log_context as log:
            try:
                with (redirect_stdout(log) if log else nullcontext()), \
                     (redirect_stderr(log) if log else nullcontext()):
                    result = operation()
                    if result.get("valid") is False:
                        raise SubmissionError(result.get("error", f"{name} failed"))
            except Exception as exc:
                if output_dir is not None:
                    emit_result({"valid": False, "reward": 0.0, "error": str(exc),
                                 **review_metadata()}, output_dir / f"{name}.json")
                    log.write(f"{type(exc).__name__}: {exc}\n")
                raise
        results[name] = result
        if output_dir is not None:
            emit_result(result, output_dir / f"{name}.json")
        return result

    stage("check", lambda: inspect_submission(
        submission, float(profile["size_limit_gib"]), tokenizer_dir=profile["tokenizer"]))
    if package_only:
        return results
    with evaluation_package(submission, profile["tokenizer"]) as package:
        stage("smoke", lambda: {"valid": True, "generated_text": generate_smoke(
            package, profile["tokenizer"], "Complete with one city name: The capital of France is",
            16, timeout), **review_metadata()})

        # Individual PPL commands and the full pipeline share measurement and reporting.
        stage("ppl", lambda: measure_quality(package, profile["tokenizer"], profile["ppl_assets"],
                                            timeout, profile["ppl_split"]))
        samples = output_dir / "decode_requests.jsonl" if output_dir is not None else None
        stage("decode", lambda: measure_decode(package, profile, timeout, samples_output=samples))
    if profile.get("score"):
        stage("score", lambda: score_measurements(results["ppl"], results["decode"], profile, public))
    return results


def run_verification(args: argparse.Namespace) -> None:
    """Hidden output adapter: use the shared pipeline and publish Harbor reward."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_path = output_dir / "reward.txt"
    reward_path.write_text("0\n", encoding="utf-8")
    profile = dict(tokenizer=args.assets / "tokenizer",
                   size_limit_gib=args.limit_gib, score=True,
                   ppl_assets=args.assets, ppl_split="hidden", decode_assets=args.assets)
    try:
        results = evaluate_submission(args.submission, profile, output_dir=output_dir)
        reward = results["score"]["reward"]
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise SubmissionError("invalid final reward")
    except Exception as exc:
        reward = 0.0
        emit_result({"valid": False, "reward": reward, "error": str(exc), **review_metadata()},
                    output_dir / "score.json")
        with (output_dir / "score.log").open("a", encoding="utf-8") as log:
            log.write(f"{type(exc).__name__}: {exc}\n")
    reward_path.write_text(f"{reward}\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the complete hidden verification pipeline")
    run.add_argument("submission", type=Path)
    run.add_argument("--assets", type=Path, default=Path(__file__).parent / "evaluator-assets")
    run.add_argument("--output-dir", type=Path, default=Path("/logs/verifier"))
    run.add_argument("--limit-gib", type=float, default=DEFAULT_LIMIT_GIB)

    check = subparsers.add_parser("check", help="Check package files, symlinks, and total size")
    check.add_argument("submission", type=Path)
    check.add_argument("--limit-gib", type=float, default=DEFAULT_LIMIT_GIB)
    check.add_argument("--output", type=Path)

    smoke = subparsers.add_parser("smoke", help="Run submitted generation offline")
    smoke.add_argument("submission", type=Path)
    smoke.add_argument("--limit-gib", type=float, default=DEFAULT_LIMIT_GIB)
    smoke.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    smoke.add_argument(
        "--prompt", default="Complete with one city name: The capital of France is"
    )
    smoke.add_argument("--max-new-tokens", type=int, default=16)
    smoke.add_argument("--timeout", type=int, default=600)
    smoke.add_argument("--output", type=Path)

    ppl = subparsers.add_parser("ppl", help="Compute fixed C4 256x2048 perplexity offline")
    ppl.add_argument("submission", type=Path)
    add_ppl_arguments(ppl)
    ppl.add_argument("--limit-gib", type=float, default=DEFAULT_LIMIT_GIB)
    ppl.add_argument("--output", type=Path)

    decode = subparsers.add_parser('decode', help='Measure first-to-last token latency after a valid PPL measurement')
    decode.add_argument('submission', type=Path)
    decode.add_argument('--ppl-result', type=Path, required=True)
    decode.add_argument('--samples-output', type=Path)
    decode.add_argument('--limit-gib', type=float, default=DEFAULT_LIMIT_GIB)
    decode.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    decode.add_argument('--timeout', type=int, default=600)
    decode.add_argument('--output', type=Path)

    score = subparsers.add_parser('score', help='Compute C4 PPL times decode reward; no GPU')
    score.add_argument('--ppl-result', type=Path, required=True)
    score.add_argument('--decode-result', type=Path, required=True)
    score.add_argument('--prerequisites-failed', action='store_true')
    score.add_argument('--output', type=Path)

    return parser


def emit_result(result: dict[str, object], output: Path | None) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(payload, end="")
    else:
        output.write_text(payload, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_verification(args)
        return
    try:
        if args.command == "check":
            result = inspect_submission(args.submission, args.limit_gib)
        elif args.command == "smoke":
            result = run_smoke(args)
        elif args.command == "ppl":
            if not math.isfinite(args.inference_timeout) or args.inference_timeout <= 0:
                raise SubmissionError("inference-timeout must be finite and positive")
            result = evaluate_ppl(args)
        elif args.command == 'decode':
            if args.timeout <= 0:
                raise SubmissionError('timeout must be positive')
            result = evaluate_decode(args)
        elif args.command == 'score':
            if args.prerequisites_failed:
                raise SubmissionError('a required check/PPL/decode stage failed; no stale results accepted')
            quality = json.loads(args.ppl_result.read_text())
            decode = json.loads(args.decode_result.read_text()) if args.decode_result.is_file() else {}
            result = final_score(quality, decode)
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except (SubmissionError, InferenceError, ValueError, OSError, OverflowError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        emit_result({"valid": False, "reward": 0.0, "error": str(exc), **review_metadata()}, args.output)
        raise SystemExit(1) from exc
    emit_result({**result, **review_metadata()}, args.output)


if __name__ == "__main__":
    main()
