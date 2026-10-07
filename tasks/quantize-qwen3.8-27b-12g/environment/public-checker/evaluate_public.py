#!/usr/bin/env python3
"""Public output adapter; all measurement stages are implemented in evaluate.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate as shared
from inference_runtime import review_metadata

ASSETS_DIR = Path(__file__).parent / "evaluator-assets"
PROFILE_CONFIG = ASSETS_DIR / "evaluation_profiles.json"
CheckError = shared.SubmissionError


def load_profile(name="formal", config_path=PROFILE_CONFIG):
    """Read task-owned settings; relative resource paths are relative to the config."""
    config_path = Path(config_path)
    profiles = json.loads(config_path.read_text(encoding="utf-8"))
    profile = dict(profiles[name])
    for key in ("tokenizer", "decode_assets", "ppl_assets"):
        path = Path(profile[key])
        profile[key] = path if path.is_absolute() else config_path.parent / path
    return profile

DEFAULT_PROFILE = load_profile()


def evaluate(submission, *, package_only=False, timeout=600, profile=None,
             quality_only=False, questions=32):
    """Select public/debug resources, call the shared pipeline, and format the response."""
    profile = DEFAULT_PROFILE if profile is None else profile
    if quality_only:
        if package_only:
            raise ValueError("quality-only and package-only are mutually exclusive")
        from quality_check import measure
        if type(questions) is not int or questions not in (32, 64):
            raise ValueError("quality questions must be 32 or 64")
        checked = shared.inspect_submission(submission, float(profile["size_limit_gib"]),
                                            tokenizer_dir=profile["tokenizer"])
        with shared.evaluation_package(submission, profile["tokenizer"]) as package:
            diagnostic = measure(package, profile["tokenizer"], questions=questions, timeout=timeout)
        return {**checked, **review_metadata(), "valid": True, "public_only": True,
                "diagnostic_only": True, "is_official_score": False,
                "formal_reward_unchanged": True, "downstream_quality": diagnostic,
                **({"debug_only": True} if not profile.get("score") else {})}
    measured = shared.evaluate_submission(submission, profile, public=True,
                                          package_only=package_only, timeout=timeout)
    result = {"valid": True, "public_only": True, **measured["check"], **review_metadata()}
    if not profile.get("score"):
        result.update(debug_only=True, is_official_score=False)
    if package_only:
        return result
    quality, speed = measured["ppl"], measured["decode"]
    result.update(generated_text=measured["smoke"]["generated_text"],
                  public_ppl=quality["ppl"], quality=quality, decode=speed)
    result.update({key: quality[key] for key in
                   ("ppl_protocol", "ppl_split", "predicted_tokens")})
    if profile.get("score"):
        cfg = json.loads((profile["ppl_assets"] / "ppl_config.json").read_text())
        result.update(public_bf16_ppl=cfg.get("bf16_ppl"),
                      public_reference_ppl=cfg.get("reference_ppl"), baseline_status=cfg["baseline_status"],
                      target_met=False)
        score = measured["score"]
        result.update(quality_gate_enabled=score["quality_gate_enabled"], public_quality_pass=score["quality_pass"])
        result["public_reward"] = score["reward"]
        for key in ("scoring_version", "decode_latency_seconds", "ppl_latency_product", "reward_mapping", "ranking_eligible", "quality_bf16_ppl", "quality_max_ratio", "quality_ppl_limit", "ppl_increase_percent", "quality_rejection_reason",
                    "quality_gray_start_ratio", "quality_gray_start_ppl",
                    "quality_max_penalty_multiplier", "quality_penalty_multiplier",
                    "penalized_ppl_latency_product"):
            result[key] = score[key]
    else:
        latency = speed["median_round_mean_seconds"]
        result.update(decode_latency_seconds=latency, ppl_latency_product=quality["ppl"] * latency)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quality-only", action="store_true",
                      help="optional MATH diagnostic only; no PPL, speed or reward")
    parser.add_argument("--questions", type=int, choices=(32, 64), default=None)
    mode.add_argument("--package-only", action="store_true",
                        help="check package files and size without loading the model")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--profile", default="formal", help="profile name in the task-owned configuration")
    parser.add_argument("--config", type=Path, default=PROFILE_CONFIG)
    args = parser.parse_args()
    if args.questions is not None and not args.quality_only:
        parser.error("--questions requires --quality-only")
    result = {"valid": False, "public_only": True, **review_metadata()}
    try:
        profile = load_profile(args.profile, args.config)
        if not profile.get("score"):
            result.update(debug_only=True, is_official_score=False)
        result = evaluate(args.submission, package_only=args.package_only,
                          timeout=args.timeout, profile=profile,
                          quality_only=args.quality_only, questions=args.questions or 32)
    except Exception as exc:
        result.update(error=str(exc), **review_metadata())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
