# SpinQuant

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary,
Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian and Tijmen Blankevoort.
*SpinQuant: LLM Quantization with Learned Rotations* (ICLR 2025).
[Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html) ·
[Implementation](https://github.com/facebookresearch/SpinQuant).

- **Idea:** paired rotations can preserve the full-precision computation while
  spreading outliers, making weights and activations easier to quantize. Optimize
  the rotations rather than relying only on a random choice.
- **Read:** section 3, rotation invariance, rotation placement and optimization;
  the experiments separating weight, activation and KV-cache precision.
- **Possible experiment:** compare no rotation, a compatible fixed rotation and
  learned rotation, accounting for any runtime transform cost.
- **Implementation work:** calibration-based rotation optimization, correct weight
  fusion and any remaining online transforms. Work through the actual attention,
  normalization and MLP equations before transferring Llama rotation placements
  to a different architecture. Match rotations to the actual submitted architecture and dimensions.

## Prepared environment and limits

Use `/opt/spinquant/bin/python`. Bundled author source is in [upstream/](upstream/),
with its CC-BY-NC-4.0 license unchanged. Dependencies isolate Transformers 4.44.2
from the inference runtime; Torch is shared with the image. The Hadamard extension
is built from pinned full source against the image's Torch/CUDA ABI.

Offline component check:
`/opt/spinquant/bin/python /opt/quant-docs/common/research_check.py spinquant`.
Author PTQ CLI: `/opt/spinquant/bin/python /opt/quant-docs/common/run_author.py spinquant ptq -- --help`.
The training entry is `spinquant train`. Actual PTQ/training must be launched via
`/opt/spinquant/bin/python -m torch.distributed.run --standalone --nproc_per_node=1`
followed by the wrapper path and arguments: the author main initializes NCCL.
These are author interfaces, not a
turnkey task solution. Their Llama walker and distributed trainer are not current
Qwen adapters. Replace the author's dataset access with local public training
blocks, keep model paths local and outputs in `/app/work`. Optional paper benchmark
packages/data are not provided; use the task Checker. Component checks do not
validate a full rotation-training run, compact export or final decode backend.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../references/papers.json)

No ready-to-run task adapter is provided; see the implementation/adaptation notes above.

[Tool availability](../../runtime/tools.md)
