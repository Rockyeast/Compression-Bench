import logging
import os
from datetime import datetime

import transformers

from d2quant import cli, config, data, evaluation, model_utils
from d2quant.quantization import gptq, pipeline
from d2quant.rotation import transforms


def setup_logger(log_file):
    logger = logging.getLogger("d2quant")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def evaluate_zero_shot(model, tokenizer, args, logger):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    wrapped_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.lm_eval_batch_size,
    )
    scores = {}
    for task in args.tasks:
        result = lm_eval.simple_evaluate(
            model=wrapped_model,
            tasks=[task],
            batch_size=args.lm_eval_batch_size,
        )["results"][task]
        score = result.get("acc_norm,none", result.get("acc,none"))
        scores[task] = round(score * 100, 2)
        logger.info("%-16s %.2f", task, scores[task])
    scores["average"] = round(sum(scores.values()) / len(scores), 2)
    logger.info("%-16s %.2f", "average", scores["average"])
    return scores


def main():
    args = cli.parse_args()
    transformers.set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.normpath(args.model))
    run_name = (
        f"{model_name}_w{args.w_bits}_g{args.w_groupsize}"
        f"_dsq{int(args.dsq)}_dac{int(args.dac)}_{timestamp}"
    )
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, f"{run_name}.log"))
    logger.info(
        "D²Quant | model=%s bits=%d group=%d DSQ=%s DAC=%s backend=%s",
        model_name,
        args.w_bits,
        args.w_groupsize,
        args.dsq,
        args.dac,
        args.backend,
    )

    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    if args.rotate:
        logger.info("Applying groupwise %s rotation", args.rotate_mode)
        transforms.fuse_layer_norms(model, args)
        transforms.rotate_model(model, args)
        config.cleanup_memory(verbose=False)

    if args.w_bits < 16:
        calibration_loader = data.get_loaders(
            args.cal_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        if args.backend == "gptaq":
            pipeline.quantize_model(
                model,
                calibration_loader,
                config.DEV,
                args,
                logger,
            )
        elif args.backend == "gptq":
            gptq.gptq_fwrd(model, calibration_loader, config.DEV, args)
        else:
            gptq.rtn_fwrd(model, config.DEV, args)

    if args.distribute:
        config.distribute_model(model)
    else:
        model.to(config.DEV)

    ppl_results = {}
    if args.ppl_eval:
        for dataset_name in ("wikitext2", "c4"):
            test_loader = data.get_loaders(
                dataset_name,
                seed=args.seed,
                model=args.model,
                seqlen=2048,
                hf_token=args.hf_token,
                eval_mode=True,
            )
            ppl_results[dataset_name] = evaluation.ppl_eval(model, test_loader)
        logger.info(
            "Perplexity | WikiText2 %.4f | C4 %.4f",
            ppl_results["wikitext2"],
            ppl_results["c4"],
        )

    if args.lm_eval:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            use_fast=False,
            token=args.hf_token,
        )
        evaluate_zero_shot(model, tokenizer, args, logger)


if __name__ == "__main__":
    main()
