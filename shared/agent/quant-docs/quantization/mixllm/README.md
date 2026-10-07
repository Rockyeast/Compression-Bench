# MixLLM

Zhen Zheng, Xiaonan Song and Chuanjie Liu. *MixLLM: LLM Quantization with Global
Mixed-precision between Output-features and Highly-efficient System Design*
(MLSys 2026).
[Paper](https://proceedings.mlsys.org/paper_files/paper/2026/hash/a66caa1703fe34705a4368c3014c1966-Abstract-Conference.html) ·
[Implementation](https://github.com/microsoft/MixLLM).

- **Idea:** assign higher precision to important output channels across the model,
  then co-design packing and kernels to make mixed precision efficient.
- **Read:** global salience identification, mixed-precision configuration,
  two-step dequantization, data conversion and the software pipeline.
- **Possible experiment:** compare a mixed-precision candidate's quality, actual
  saved size and measured decode latency against a uniform-precision candidate.
- **Implementation work:** calibration, output-channel allocation, tensor packing
  and a matching kernel/loader. The public example mixes INT4/INT8 weights with
  INT8 activations; INT2/INT3 variants require additional implementation.
  The repository's vLLM patch targets 0.9.0; this environment uses 0.28.0.
  Distinguish its fake-quantization evaluation from real packed inference.
  Large-batch throughput gains must be remeasured for this task's decode workload.

## Availability: source only; runtime not installed

Pinned MIT-licensed author code is bundled under [upstream/](upstream/), including
its vLLM patch for inspection. The patch targets vLLM 0.9.0; the benchmark fixes
0.28.0. Its CUDA Makefile additionally hard-codes `/opt/conda`, Python 3.11 and
SM80; our image uses Python 3.12. Installing it unchanged would not supply a valid
current runtime. The CUTLASS/vLLM submodules are not bundled, and the author's
`pip install`/patch command is not executed during task setup.

This is an explicit integration blocker, not a request that the Agent install
missing packages during a timed task. Port the build and inference interface and
validate a packed layer before promoting this method to an executable tool.
The existing AutoRound mixed-bit tool remains available, but it does not implement
MixLLM's output-channel algorithm or kernels.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../references/papers.json)

No ready-to-run task adapter is provided; see the implementation/adaptation notes above.

[Tool availability](../../runtime/tools.md)
