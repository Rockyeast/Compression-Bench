#!/usr/bin/env python3
"""Restore the pinned offline library; existing valid files need no download."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'shared/agent/reference-lock.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError(f'Unsafe reference path: {name}')
    target = root.joinpath(*p.parts)
    # Materialized references are files, never symlinks (including parents).
    if any(part.is_symlink() for part in [target, *target.parents]):
        raise ValueError(f'Symlink in reference destination: {name}')
    return target


def matches(path, entry):
    return (path.is_file() and not path.is_symlink()
            and path.stat().st_size == entry['bytes']
            and stat.S_IMODE(path.stat().st_mode) & 0o111 == entry['mode'] & 0o111
            and digest(path) == entry['sha256'])


def load_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported reference manifest version')
    seen = set()
    names = set()
    for pack in manifest['archives']:
        if Path(pack['name']).name != pack['name'] or pack['name'] in names:
            raise ValueError('Invalid or duplicate archive name')
        names.add(pack['name'])
        for entry in pack['files']:
            name = entry['path']
            safe_path(Path('/reference-validation-root'), name)
            if name in seen or entry['mode'] not in (0o644, 0o755):
                raise ValueError(f'Invalid or duplicate reference entry: {name}')
            seen.add(name)
    return manifest


def archive_file(pack, manifest, cache, archive_dir=None):
    path = (archive_dir or cache) / pack['name']
    if path.is_file() and path.stat().st_size == pack['bytes'] and digest(path) == pack['sha256']:
        return path
    if archive_dir:
        raise ValueError(f'Missing or corrupt local archive: {path}')
    base = manifest.get('download_base_url')
    if not base:
        raise ValueError('No published download URL; supply --archive-dir with the prepared archives')
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, prefix='.download-', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            request = urllib.request.Request(base.rstrip('/') + '/' + pack['name'],
                                             headers={'User-Agent': 'Compression-Bench-reference-fetcher'})
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, stream)
            stream.close()
            if tmp.stat().st_size != pack['bytes'] or digest(tmp) != pack['sha256']:
                raise ValueError(f'Archive checksum mismatch: {pack["name"]}')
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return path


def restore_archive(archive, pack, destination, needed):
    expected = {e['path']: e for e in pack['files']}
    seen = set()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reference-stage-', dir=destination.parent) as tmp:
        stage = Path(tmp)
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                if member.name not in expected or member.name in seen or not member.isfile():
                    raise ValueError(f'Unexpected archive member: {member.name}')
                entry = expected[member.name]
                if member.size != entry['bytes'] or member.mode != entry['mode']:
                    raise ValueError(f'Archive metadata mismatch: {member.name}')
                target = safe_path(stage, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('wb') as out:
                    shutil.copyfileobj(src, out)
                target.chmod(entry['mode'])
                if not matches(target, entry):
                    raise ValueError(f'File checksum mismatch: {member.name}')
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError('Archive is missing manifest entries')
        for name in needed:
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(safe_path(stage, name), target)


def prepare(manifest_path, destination, cache, archive_dir=None, verify=False, repair=False):
    manifest = load_manifest(manifest_path)
    pending = []
    changed = []
    total = 0
    for pack in manifest['archives']:
        needed = []
        for entry in pack['files']:
            total += 1
            target = safe_path(destination, entry['path'])
            if not matches(target, entry):
                needed.append(entry['path'])
                if target.exists():
                    changed.append(entry['path'])
        if needed:
            pending.append((pack, needed))
    if verify and pending:
        raise ValueError(f'{sum(len(n) for _, n in pending)} reference files are missing or changed')
    if changed and not repair:
        raise ValueError(f'{len(changed)} reference files differ; preserve your edits or explicitly use --repair. First: {changed[0]}')
    for pack, needed in pending:
        print(f'Restoring {len(needed)} files from {pack["name"]}', flush=True)
        archive = archive_file(pack, manifest, cache, archive_dir)
        restore_archive(archive, pack, destination, needed)
    print(f'References verified: {total} files; restored {sum(len(n) for _, n in pending)}.', flush=True)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--destination', type=Path, default=ROOT / 'shared/agent/quant-docs')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.cache/quantbench/references')
    parser.add_argument('--archive-dir', type=Path, help='Use already downloaded archives, entirely offline')
    parser.add_argument('--verify', action='store_true', help='Check all files without downloading or writing')
    parser.add_argument('--repair', action='store_true', help='Replace modified reference files with the pinned originals')
    args = parser.parse_args()
    try:
        prepare(args.manifest, args.destination, args.cache_dir, args.archive_dir, args.verify, args.repair)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f'Reference preparation failed: {error}\n')


if __name__ == '__main__':
    main()
