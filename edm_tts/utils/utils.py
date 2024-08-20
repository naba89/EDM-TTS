import logging
import os
import sys

import torch

import transformers
from transformers.trainer_utils import is_main_process, get_last_checkpoint


def setup_logging(logger, training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.ERROR if is_main_process(training_args.local_rank) else logging.ERROR)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)


def detect_last_checkpoint(logger, training_args):
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return last_checkpoint


def random_topk_mask(mask_len, probs, distribution, temperature=1.0):
    # sort the probs and get the threshold for the top-k tokens
    log_probs = torch.log(probs)
    gumbel_noise = distribution.sample(probs.shape).squeeze(-1)
    confidence = log_probs + temperature * gumbel_noise
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = torch.take_along_dim(sorted_confidence, mask_len.long().unsqueeze(-1), dim=-1)

    # Masks tokens with lower confidence.
    mask = confidence < cut_off
    return mask
