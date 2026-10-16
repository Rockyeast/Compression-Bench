"""Optional public Checker math diagnostic; excluded from reward and Verifier."""
from __future__ import annotations

import json
from pathlib import Path
import time

from inference_runtime import SubmittedRuntime

QUALITY_MAX_CONCURRENCY = 256

DATA_PATH = Path(__file__).resolve().parent / "evaluator-assets" / "math500_probe.json"
PROMPT_SUFFIX = r"Please reason step by step, and put your final answer within \boxed{}."


def load_questions(questions):
    if type(questions) is not int or questions not in (32, 64):
        raise ValueError("quality questions must be 32 or 64")
    raw = DATA_PATH.read_bytes()
    payload = json.loads(raw)
    if len(payload["questions"]) != 64:
        raise ValueError("expected a fixed nested 64-question set")
    return payload, payload["questions"][:questions]


def prompt_ids(tokenizer, problem):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem + "\n\n" + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def measure(submission, tokenizer_dir, *, questions=32, timeout=600):
    from math_verify import parse, verify
    from transformers import AutoTokenizer

    payload, rows = load_questions(questions)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    prompts = [prompt_ids(tokenizer, row["problem"]) for row in rows]
    max_new_tokens, max_model_len = 8192, 12288
    if any(len(ids) + max_new_tokens > max_model_len for ids in prompts):
        raise ValueError("math prompt exceeds context budget")
    started = time.monotonic()
    items = []
    with SubmittedRuntime(submission, max_model_len=max_model_len,
                          max_num_seqs=min(QUALITY_MAX_CONCURRENCY, len(prompts)),
                          gpu_memory_utilization=0.8, timeout=timeout) as engine:
        # Queue all questions together; vLLM refills active slots as answers finish.
        outputs = engine.generate(prompts, max_new_tokens=max_new_tokens,
                                  vocab_size=len(tokenizer), ignore_eos=False)
        for row, output in zip(rows, outputs, strict=True):
            text = tokenizer.decode(output, skip_special_tokens=True)
            gold = parse(r"\boxed{" + row["answer"] + "}")
            prediction = parse(text)
            items.append(dict(id=row["id"], level=row["level"], subject=row["subject"],
                              correct=bool(gold and prediction and verify(gold, prediction)),
                              generated_tokens=len(output), truncated=len(output) >= max_new_tokens))
    correct = sum(item["correct"] for item in items)
    return dict(dataset=payload["dataset"], dataset_revision=payload["revision"],
                questions=len(items), correct=correct,
                accuracy=correct / len(items),
                elapsed_seconds=time.monotonic() - started,
                truncated=sum(item["truncated"] for item in items),
                max_new_tokens=max_new_tokens, max_model_len=max_model_len,
                batch_size=min(QUALITY_MAX_CONCURRENCY, len(prompts)),
                scheduling="queued", enable_thinking=False,
                is_official_score=False, formal_reward_unchanged=True, items=items)
