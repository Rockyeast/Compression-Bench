"""Keepalive and health probe for the task's single-copy submission lock."""
import argparse
import fcntl
from pathlib import Path
import signal

LOCK = Path('/run/quantization-bench-submission.lock')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['hold', 'ready'])
    parser.add_argument('--lock', type=Path, default=LOCK)
    args = parser.parse_args()
    # Bind-mounted read-only: Agent cannot replace the lock inode. LOCK_EX on
    # a read-only fd works with Linux flock (local Docker host filesystem).
    with args.lock.open('rb') as held:
        if args.mode == 'ready':
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            return 1
        fcntl.flock(held, fcntl.LOCK_SH | fcntl.LOCK_NB)
        # This is main's keepalive / PID 1. Exiting it stops the container.
        signal.signal(signal.SIGTERM, lambda *_: exit(0))
        signal.signal(signal.SIGINT, lambda *_: exit(0))
        while True:
            signal.pause()


if __name__ == '__main__':
    raise SystemExit(main())
