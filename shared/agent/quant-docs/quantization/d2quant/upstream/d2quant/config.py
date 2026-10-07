import gc
import inspect
import logging

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from . import model_utils

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cleanup_memory(verbose=True, verbos=None):
    """Collect Python objects and release unused CUDA memory."""
    if verbos is not None:
        verbose = verbos

    caller = ""
    try:
        caller = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def reserved_memory():
        return sum(
            torch.cuda.memory_reserved(device=index)
            for index in range(torch.cuda.device_count())
        )

    before = reserved_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        after = reserved_memory()
        if verbose:
            logging.debug(
                "GPU memory%s: %.2f -> %.2f GB (%.2f GB)",
                caller,
                before / 1024**3,
                after / 1024**3,
                (after - before) / 1024**3,
            )


def distribute_model(model):
    """Distribute a model across all visible GPUs for evaluation."""

    model_type = model_utils.get_model_type(model)
    module_class = {
        model_utils.LLAMA_MODEL: "LlamaDecoderLayer",
        model_utils.QWEN3_MODEL: "Qwen3DecoderLayer",
        model_utils.QWEN2_MODEL: "Qwen2DecoderLayer",
    }.get(model_type)
    if module_class is None:
        raise ValueError(f"Distributed evaluation is unsupported for {model_type}")

    no_split = [module_class]
    max_memory = get_balanced_memory(model, no_split_module_classes=no_split)
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split,
    )
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )
    cleanup_memory()
