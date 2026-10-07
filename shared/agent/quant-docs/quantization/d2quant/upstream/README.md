# D<sup>2</sup>Quant: Accurate Low-bit Post-Training Weight Quantization for LLMs

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2026-blue">
  <a href="https://arxiv.org/abs/2602.02546">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoSvg">
  </a>
  <a href="https://github.com/XIANGLONGYAN/D2Quant">
    <img src="https://img.shields.io/github/stars/XIANGLONGYAN/D2Quant?style=social">
  </a>
  <a href="https://github.com/XIANGLONGYAN/D2Quant">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=XIANGLONGYAN.D2Quant&right_color=violet">
  </a>
</p>

[Xianglong Yan](https://xianglongyan.github.io/), Chengzhu Bao,
[Zhiteng Li](https://zhitengli.github.io),
[Tianao Zhang](https://zta20040910.github.io/),
[Shaoqiu Zhang](https://qiushao-e.github.io/),
[Ruobing Xie](https://ruobingxie.github.io/), Xingwu Sun, and
[Yulun Zhang](http://yulunzhang.com/),
"D²Quant: Accurate Low-bit Post-Training Weight Quantization for LLMs", NeurIPS 2026

---

#### 🔥 News

- **NeurIPS 2026:** D²Quant has been accepted to NeurIPS 2026.
- **2026-07-11:** The D²Quant code is released.
- **2026-01-30:** This repository is released.

---

> **Abstract:** Large language models (LLMs) deliver strong performance, but their high compute and memory costs make deployment difficult in resource-constrained scenarios. Weight-only post-training quantization (PTQ) is appealing, as it reduces memory usage and enables practical speedup without low-bit operators or specialized hardware. However, accuracy often degrades significantly in weight-only PTQ at sub-4-bit precision, and our analysis identifies two main causes: (1) down-projection matrices are a well-known quantization bottleneck, but maintaining their fidelity often requires extra bit-width; (2) weight quantization induces activation deviations, but effective correction strategies remain underexplored. To address these issues, we propose D²Quant, a novel weight-only PTQ framework that improves quantization from both the weight and activation perspectives. On the weight side, we design a Dual-Scale Quantizer (DSQ) tailored to down-projection matrices, with an absorbable scaling factor that significantly improves accuracy without increasing the bit budget. On the activation side, we propose Deviation-Aware Correction (DAC), which incorporates a mean-shift correction within LayerNorm to mitigate quantization-induced activation distribution shifts. Extensive experiments across multiple LLM families and evaluation metrics show that D²Quant delivers superior performance for weight-only PTQ at sub-4-bit precision.

<p align="center">
  <img width="100%" src="figs/overview.png">
</p>

---

## 🔗 Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#-results)
- [Citation](#citation)
- [Acknowledgements](#-acknowledgements)

## Installation

```bash
git clone --recurse-submodules https://github.com/XIANGLONGYAN/D2Quant.git
cd D2Quant

conda create -n d2quant python=3.10 -y
conda activate d2quant
pip install -r requirements.txt
pip install -e fast-hadamard-transform --no-build-isolation
```

The CUDA toolkit used to build `fast-hadamard-transform` must match the CUDA
version of the installed PyTorch package.

## Data Preparation

Calibration and perplexity datasets are loaded from `$D2QUANT_DATA_ROOT`
(default: `./data`).

```bash
python prepare_data.py --data_root ./data --datasets wikitext,ptb,c4
```

## Usage

The release CLI defaults to the complete D²Quant configuration: asymmetric
GPTAQ, groupwise Hadamard rotation, DSQ, and DAC.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model /path/to/Meta-Llama-3-8B \
    --bits 2 \
    --group-size 128 \
    --dsq \
    --dac \
    --ppl-eval
```

Additional options:

- `--bits {2,3,16}`: weight bit-width.
- `--backend {gptaq,gptq,rtn}`: quantization backend.
- `--num-samples`: number of calibration samples (default: 128).
- `--num-iters`: DSQ fitting iterations (default: 15).
- `--lm-eval`: run zero-shot evaluation.

See `python main.py --help` and `run.sh` for details.

Run the lightweight DSQ / DAC component tests with:

```bash
python -m unittest discover -s tests -v
```

## Code Structure

```text
main.py                  # quantization and evaluation entry point
prepare_data.py          # dataset preparation
run.sh                   # example commands
d2quant/
├── cli.py               # release command-line interface
├── config.py            # runtime / multi-GPU helpers
├── data.py              # calibration and evaluation data loaders
├── evaluation.py        # perplexity evaluation
├── model_utils.py       # model-family abstractions and calibration hooks
├── quantization/
│   ├── pipeline.py      # layerwise GPTAQ + DSQ + DAC pipeline
│   ├── gptq.py          # GPTQ / RTN baselines
│   ├── quantizer.py     # standard weight quantizer
│   ├── dsq_2bit.py      # 2-bit Dual-Scale Quantizer
│   ├── dsq_3bit.py      # 3-bit Dual-Scale Quantizer
│   ├── dac.py           # Deviation-Aware Correction
│   ├── diagnostics.py   # calibration diagnostics
│   └── gar.py           # Hessian-guided group ordering
└── rotation/
    ├── transforms.py    # groupwise model rotations
    └── hadamard.py      # Hadamard / Walsh / DCT utilities
```

## 🔎 Results

<details>
<summary>D<sup>2</sup>Quant demonstrates superior performance on the Qwen-3 model series under 2-bit weight-only quantization. (click to expand)</summary>
<p align="center">
  <img width="100%" src="figs/table1.png">
</p>
</details>

<details>
<summary>D<sup>2</sup>Quant demonstrates superior performance on the LLaMA-3 and LLaMA-3.1 model series under 2-bit weight-only quantization. (click to expand)</summary>
<p align="center">
  <img width="100%" src="figs/table2.png">
</p>
</details>

## Citation

If you find the code helpful in your research or work, please cite:

```bibtex
@inproceedings{yan2026d,
  title={D$^2$Quant: Accurate Low-bit Post-Training Weight Quantization for LLMs},
  author={Yan, Xianglong and Bao, Chengzhu and Li, Zhiteng and Zhang, Tianao and Zhang, Shaoqiu and Xie, Ruobing and Sun, Xingwu and Zhang, Yulun},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://arxiv.org/abs/2602.02546}
}
```

## 💡 Acknowledgements

This work is released under the [Apache 2.0 License](LICENSE). The code builds
on [QuaRot](https://github.com/spcl/QuaRot),
[GPTQ](https://github.com/IST-DASLab/gptq), and GPTAQ.
