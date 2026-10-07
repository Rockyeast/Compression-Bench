# D2Quant

Xianglong Yan, Chengzhu Bao, Zhiteng Li, Tianao Zhang, Shaoqiu Zhang, Ruobing Xie,
Xingwu Sun and Yulun Zhang. *D²Quant: Accurate Low-bit Post-Training Weight
Quantization for LLMs* (2026 preprint).
[Paper](https://arxiv.org/abs/2602.02546v2) ·
[Implementation](https://github.com/XIANGLONGYAN/D2Quant).

- **Idea:** improve a sensitive down-projection using two scaling granularities,
  and correct mean shifts in intermediate activations caused by quantized weights.
- **Read:** sections 3.2 Dual-Scale Quantizer and 3.3 Deviation-Aware Correction;
  the ablations separating their contributions.
- **Possible experiment:** test down-projection scaling separately from activation
  correction before combining them, at fixed storage precision.
- **Implementation work:** calibrated quantization, paired up/down scale absorption
  and, for the correction component, matching normalization behavior in inference.
  Weight-only quantization can still change activation distributions. Check the
  target model's normalization and branching before implementing these transforms.

## Prepared environment and limits

The inspected author implementation is now bundled under [upstream/](upstream/)
(Apache-2.0). Use `/opt/d2quant/bin/python`, isolated Transformers 4.57.3 and the
image's shared Torch; this is a compatibility environment, not the author's exact
Torch 2.2.1 environment. The Hadamard extension is compiled for the current ABI.

Run `/opt/d2quant/bin/python /opt/quant-docs/common/research_check.py d2quant` to
check 2/3-bit dual-scale reconstruction and mean-shift correction. The offline
author entry is `/opt/d2quant/bin/python /opt/quant-docs/common/run_author.py d2quant quantize -- --help`.
The author loader handles Llama, OPT and older Qwen2/Qwen3 structures; current
hybrid models require adaptation. Author calibration uses `D2QUANT_DATA_ROOT`
with dataset directories created by `datasets.save_to_disk`; adapt it to the
provided public training split. Do not run `prepare_data.py` to download data
inside a task. Optional `lm-eval` benchmarks are omitted; use the Checker.

The author quantizers return reconstructed floating-point tensors. The released
CLI does not provide this benchmark's packed submission/export/load pipeline.
Calibration success therefore does not prove reduced package size or faster
inference. In particular DAC changes normalization forward behavior: exporting
weights without preserving that computation is incorrect.

## Files and availability

[Paper PDF](paper.pdf) · [Searchable text](paper.txt) · [Provenance](../../references/papers.json)

No ready-to-run task adapter is provided; see the implementation/adaptation notes above.

[Tool availability](../../runtime/tools.md)
