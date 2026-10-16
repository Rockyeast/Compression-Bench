# Preparing the offline reference library

Git stores method guides, project-owned tools, source provenance and `reference-lock.json`.
Papers, extracted paper text in the paper collections, and selected upstream sources are
stored in immutable, checksummed `.tar.gz` archives. The prepared Agent image retains
these files under `/opt/quant-docs/`, with the same paths and executable permissions.

## Current distribution

The fixed archive set is available from the [reference library release](https://github.com/Rockyeast/Compression-Bench/releases/tag/reference-library-eced629bf1eb).
The lock file pins its download URL and checksums. The build entry point downloads
missing archives automatically; existing verified reference files need no download.
For manual or offline handoff, download all ten archives and keep them together.

From the repository root, restore a local archive set:

```bash
python3 scripts/prepare_references.py --archive-dir /path/to/reference-archives
```

Or let the build entry point restore it:

```bash
QUANTBENCH_REFERENCE_ARCHIVE_DIR=/path/to/reference-archives \
  bash scripts/build_agent_base.sh --build-arg CUDA_ARCH_LIST="<compute-capability>"
```

For a direct `docker build shared/agent`, run preparation first.

Check an existing library without writes or downloads:

```bash
python3 scripts/prepare_references.py --verify
```

Missing files are restored automatically. Modified files stop preparation so local
work is preserved; back up any changes before explicitly using `--repair` to restore
the pinned originals. Download/cache files live in `.cache/quantbench/references/`
when a published URL is configured; `--archive-dir` uses the supplied files in place.

## Provenance and reproducibility

`reference-lock.json` records the source Git commit and the size, SHA-256 and permissions
of every archived file. Each archive has its own SHA-256. The selected upstream trees
include the existing source selection and local adaptations; restoring a current
upstream branch would not reproduce these inputs. Original repository/commit links,
paper citations and license files remain available in the guides and restored sources.
Existing extracted text is archived as-is, avoiding changes from PDF extraction versions.

Preparation checks archive integrity before extraction, validates every member and
restores files atomically from a temporary staging directory. Unsafe paths, symlinks,
unlisted members and changed contents are rejected. Valid existing files need no download.
The archive files plus the lock file are sufficient for offline restoration.

## Maintaining and distributing archives

To regenerate archives from the currently pinned, verified library:

```bash
python3 scripts/package_references.py --output-dir /path/to/new-reference-archives
```

This preserves the file selection and checksums in the existing lock. Initial migration
from a tracked snapshot uses `--from-git COMMIT`; it checks local contents against Git
before packaging. Treat changes to the selection or contents as an explicit library
update, with a new source snapshot and lock review.

For a future library release, publish the new archives and set `download_base_url` to
that fixed release's download directory. Preparation verifies the pinned hashes.
Keep published archives immutable; use a new release for changed contents.

Moving files out of Git reduces the current source snapshot. Existing private Git
history still contains older copies. The complete offline image and the first-time
reference download remain approximately the same size; this is a repository change.
