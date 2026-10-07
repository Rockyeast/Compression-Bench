"""Run bundled author code with offline flags; use its matching /opt environment.

This does not adapt model walkers, datasets, export formats or scoring interfaces.
Copy source into /app/work if changing it; bundled source is read-only.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
from author_check import author_path

ENTRIES = {
    'spinquant': {'ptq': 'ptq.py', 'train': 'optimize_rotation.py'},
    'd2quant': {'quantize': 'main.py'},
    'qtip': {'quantize': 'quantize_llama/quantize_finetune_llama.py'},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=ENTRIES)
    p.add_argument('entry')
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.entry not in ENTRIES[a.method]:
        p.error(f'Choose entry from {sorted(ENTRIES[a.method])}')
    author_path(a.method)  # Sets offline flags before importing any ML packages.
    manifest = Path(__file__).resolve().parents[1]/'references/code.json'
    repo = next(r for r in json.loads(manifest.read_text())['repositories'] if r['name'] == a.method)
    entry = (manifest.parent/repo['directory_path']/ENTRIES[a.method][a.entry]).resolve()
    args = a.args[1:] if a.args[:1] == ['--'] else a.args
    sys.argv = [str(entry), *args]
    if a.method == 'spinquant' and any(x in args for x in ('--help', '-h')):
        # The author main initializes NCCL before parsing; help needs no GPU.
        from utils.process_args import process_args_ptq
        process_args_ptq()
        return
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
