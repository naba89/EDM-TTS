import functools
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import (
    set_seed, HfArgumentParser, TrainingArguments, TrainerCallback, Trainer, )

from edm_tts.datasets.audio_loading_utils import load_code_segments
from edm_tts.models.injection_conformer.modeling_injection_conformer import (InjectionConformerModel,
                                                                             InjectionConformerConfig)
from edm_tts.utils.utils import setup_logging, detect_last_checkpoint

logger = get_logger(__name__)


@dataclass
class ModelArguments:

    model_type: str = field(
        metadata={"help": "The type of model to use."},
    )

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    resume_from: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint to resume training from."},
    )
    extra_model_params: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Extra model parameters."},
    )

    wandb_project: str = field(
        default="edm_tts",
        metadata={"help": "Wandb project name."},
    )

    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training steps."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_args: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Arguments for loading the dataset."},
    )
    training_segment_length: float = field(
        default=15.36,
        metadata={"help": "The length of the training segments."},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading a dataset."},
    )

    preprocessing_only: bool = field(
        default=False,
        metadata={"help": "Whether to only preprocess the dataset and exit."},
    )

    time_limit: Optional[str] = field(
        default=None,
        metadata={"help": "Time limit for training. Format: hh:mm"},
    )


class EndTrainingCallback(TrainerCallback):
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_steps is not None and state.global_step >= self.max_steps:
            control.should_training_stop = True
            control.should_log = True
            control.should_save = True

        return control


class TimeLimitCallback(TrainerCallback):
    def __init__(self, time_limit_str):
        self.time_limit_str = time_limit_str
        # Parse the time limit string (format "hh:mm") and convert to seconds
        hours, minutes = map(int, time_limit_str.split(':'))
        self.time_limit_seconds = (hours * 3600) + (minutes * 60)
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit_seconds:
            print(f"Time limit reached: {self.time_limit_str}. Saving model and exiting.")
            control.should_save = True
            control.should_training_stop = True
            control.should_log = True


def main():
    # check_git_status()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2:
        if sys.argv[1].endswith(".yaml"):
            model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
        else:
            raise ValueError("Invalid file format. Only .yaml is supported.")
    elif len(sys.argv) == 3:
        if all(arg.endswith(".yaml") for arg in sys.argv[1:3]):
            model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
        else:
            raise ValueError("Invalid file format. Both arguments must be .yaml files.")
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set wandb project name before initializing wandb.
    os.environ["WANDB_PROJECT"] = model_args.wandb_project  # name your W&B project
    # os.environ["WANDB_WATCH"] = "all"  # log gradients and model parameters

    # Detecting last checkpoint.
    last_checkpoint = detect_last_checkpoint(logger, training_args)

    # Setup logging
    setup_logging(logger, training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    with training_args.main_process_first(desc="dataset loading"):
        dataset = load_dataset(**data_args.dataset_args,
                               trust_remote_code=True)
        train_dataset = dataset['train']
        train_dataset = train_dataset.shuffle(buffer_size=128)
        semantic_sample_rate = 16000
        semantic_downsample_factor = 320
        codec_sample_rate = 16000
        codec_downsample_factor = 320
        train_dataset = train_dataset.map(functools.partial(load_code_segments,
                                                            segment_length=data_args.training_segment_length,
                                                            random_segment=True,
                                                            acoustic_sample_rate=codec_sample_rate,
                                                            acoustic_downsample_factor=codec_downsample_factor,
                                                            semantic_sample_rate=semantic_sample_rate,
                                                            semantic_downsample_factor=semantic_downsample_factor),
                                          batched=True, batch_size=1,
                                          remove_columns=['id'])

    if data_args.preprocessing_only:
        return

    # initialize random model
    if model_args.model_type == "injection_conformer":
        config_class = InjectionConformerConfig
        model_class = InjectionConformerModel
    else:
        raise ValueError(f"Invalid model type: {model_args.model_type}")
    # load the base config
    config = config_class.from_pretrained(model_args.model_name_or_path)

    if model_args.extra_model_params is not None:
        config.update(model_args.extra_model_params)
    # reinitialize the config with updated dict (this is a hack)
    config = config_class(**config.to_dict())
    model = model_class(config).to(training_args.device)

    if (os.path.isdir(model_args.model_name_or_path) and
            'model.safetensors' in os.listdir(model_args.model_name_or_path)):
        # Load from a local checkpoint
        state_dict = load_file(os.path.join(model_args.model_name_or_path, "model.safetensors"))
        model.load_state_dict(state_dict, strict=True)

    callbacks = [EndTrainingCallback(model_args.max_train_steps)]
    if data_args.time_limit is not None:
        callbacks.append(TimeLimitCallback(data_args.time_limit))

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        # Resume from checkpoint if needed, resume_from has priority over last_checkpoint
        if model_args.resume_from is not None and os.path.isdir(model_args.resume_from):
            checkpoint = model_args.resume_from
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
