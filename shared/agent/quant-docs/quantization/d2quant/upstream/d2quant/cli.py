import argparse


DEFAULT_TASKS = [
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "rte",
    "openbookqa",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="D²Quant: low-bit post-training weight quantization for LLMs"
    )
    parser.add_argument(
        "--model", required=True, help="Hugging Face model ID or local path"
    )
    parser.add_argument(
        "--bits", dest="w_bits", type=int, default=2, choices=[2, 3, 16]
    )
    parser.add_argument("--group-size", dest="w_groupsize", type=int, default=128)
    parser.add_argument(
        "--backend",
        choices=["gptaq", "gptq", "rtn"],
        default="gptaq",
        help="Underlying post-training quantization backend",
    )
    parser.add_argument(
        "--dsq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Dual-Scale Quantization on down-projection layers",
    )
    parser.add_argument(
        "--dac",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Deviation-Aware Correction for LayerNorm mean shifts",
    )
    parser.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the QuaRot weight rotation used by D²Quant",
    )
    parser.add_argument(
        "--groupwise-rotation",
        dest="groupwise_rot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rotation-mode",
        dest="rotate_mode",
        choices=["hadamard", "random", "walsh", "dct"],
        default="hadamard",
    )
    parser.add_argument(
        "--asymmetric",
        dest="w_asym",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clip",
        dest="w_clip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--calibration-dataset", dest="cal_dataset", default="wikitext2"
    )
    parser.add_argument("--num-samples", dest="nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--num-iters", dest="num_iters", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-token", dest="hf_token")

    parser.add_argument("--ppl-eval", dest="ppl_eval", action="store_true")
    parser.add_argument("--lm-eval", dest="lm_eval", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--lm-eval-batch-size", type=int, default=32)
    parser.add_argument("--distribute", action="store_true")
    parser.add_argument("--log-dir", default="./log")

    args = parser.parse_args()

    # Internal compatibility defaults. These options belonged to exploratory
    # GPTQ branches and are intentionally not exposed in the release CLI.
    args.asym_calibrate = args.backend == "gptaq"
    args.w_rtn = args.backend == "rtn"
    args.disable_online_had = True
    args.int8_down_proj = False
    args.act_order = False
    args.static_groups = False
    return args
