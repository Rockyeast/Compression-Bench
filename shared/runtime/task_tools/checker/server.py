#!/usr/bin/env python3
"""Serialized Unix-socket runner for the isolated public Checker."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 65536
TOKENIZER_NOTICE = (
    "Candidate tokenizer files are not used for scoring. "
    "For calibration and local tests, use the original model's official tokenizer: "
    "/input/model for formal tasks, or /input/debug-model for debug-9b. "
    "Scores obtained with a different tokenizer are not comparable."
)


def receive_line(connection: socket.socket) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_REQUEST_BYTES:
            raise ValueError("request exceeded 64 KiB")
        if b"\n" in chunk:
            break
    return b"".join(chunks).split(b"\n", 1)[0]


def send_response(connection: socket.socket, response: dict[str, Any]) -> None:
    response = {**response, "tokenizer_notice": TOKENIZER_NOTICE}
    try:
        connection.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
    except ConnectionError as exc:
        # A disconnected client must not stop the service.
        print(f"Checker client disconnected; response not delivered: {exc}",
              file=sys.stderr, flush=True)


def run_evaluator(
    evaluator: Path,
    submission: Path,
    package_only: bool,
    timeout: int,
    profile: str = "formal",
    quality_only: bool = False,
    questions: int = 32,
) -> dict[str, Any]:
    if quality_only and (package_only or type(questions) is not int or questions not in (32, 64)):
        raise ValueError("invalid quality-only request")
    with tempfile.TemporaryDirectory(prefix="quant-checker-") as temp_dir:
        result_path = Path(temp_dir) / "result.json"
        command = [
            sys.executable,
            str(evaluator),
            str(submission),
            "--output",
            str(result_path),
            "--profile",
            profile,
        ]
        if quality_only:
            command.extend(["--quality-only", "--questions", str(questions)])
        if package_only:
            command.append("--package-only")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            result = {
                "valid": False,
                "error": "public evaluator returned invalid JSON",
            }
    if completed.returncode != 0 and result.get("valid", True):
        result = {"valid": False, "error": "public evaluator failed"}
    if not result.get("valid", False) and completed.stderr:
        result["diagnostic"] = completed.stderr[-2000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--evaluation-timeout", type=int, required=True)
    parser.add_argument("--debug-submission", type=Path, default=Path("/debug-submission"))
    args = parser.parse_args()

    if args.evaluation_timeout <= 0:
        raise SystemExit("Checker evaluation timeout must be positive")
    if not args.evaluator.is_file():
        raise SystemExit(f"public evaluator does not exist: {args.evaluator}")

    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        args.socket.unlink()

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def stop(_signum: int, _frame: object) -> None:
        listener.close()
        if args.socket.exists():
            args.socket.unlink()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    listener.bind(str(args.socket))
    os.chmod(args.socket, 0o666)
    listener.listen(1)

    full_checks = 0
    while True:
        connection, _ = listener.accept()
        with connection:
            try:
                request = json.loads(receive_line(connection))
                if not isinstance(request, dict):
                    raise ValueError("Checker request must be an object")
                if request.get("protocol") != 1:
                    raise ValueError("unsupported Checker protocol")
                action = request.get("action")
                mode = request.get("mode", "formal")
                if not isinstance(mode, str) or mode not in {"formal", "debug-9b"}:
                    raise ValueError("unsupported Checker mode")
                if not isinstance(action, str) or action not in {"package-only", "full-check", "quality-only"}:
                    raise ValueError("unsupported Checker action")
                questions = request.get("questions", 32)
                if "questions" in request and action != "quality-only":
                    raise ValueError("questions requires quality-only")
                if type(questions) is not int or questions not in (32, 64):
                    raise ValueError("quality questions must be 32 or 64")
                if action == "full-check":
                    full_checks += 1
                response = run_evaluator(
                    args.evaluator,
                    args.debug_submission if mode == "debug-9b" else args.submission,
                    action == "package-only",
                    args.evaluation_timeout,
                    profile=mode, quality_only=action == "quality-only", questions=questions,
                )
                response["full_checks_used"] = full_checks
                if mode == "debug-9b":
                    response.update(debug_only=True, is_official_score=False)
                    for key in ("reward", "public_reward", "target_met"):
                        response.pop(key, None)
                send_response(connection, response)
            except ConnectionError as exc:
                print(f"Checker client disconnected while receiving: {exc}",
                      file=sys.stderr, flush=True)
            except (ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
                send_response(connection, {"valid": False, "error": str(exc)})


if __name__ == "__main__":
    main()
