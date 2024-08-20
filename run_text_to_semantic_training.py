import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import torch
import wandb
from accelerate.logging import get_logger
from datasets import load_dataset
from transformers import (
    set_seed, HfArgumentParser, TrainingArguments, TrainerCallback, Trainer, )

from edm_tts.models.text_to_semantic.modeling_text_to_semantic import TextToSemanticWLen, TextToSemanticWLenConfig

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
        default="ut_ssl",
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
            # control.should_evaluate = True
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
            # control.should_evaluate = True
            control.should_training_stop = True


@dataclass
class DataCollator:
    special_tokens: dict = field(default_factory=lambda: {
            "pad": 0,
            "text": 1,
            "speech": 2,
            "sep": 3,
            "mask": 4,
        })
    num_special_tokens: int = field(init=False)
    num_text_tokens: int = 256
    num_speech_tokens: int = 1024

    def __post_init__(self):
        self.num_special_tokens = len(self.special_tokens)

    def __call__(self, features):

        # from list of dicts to dict of lists
        text_byte_strings = [feature["transcription_bytes"] for feature in features]
        semantic_tokens = [feature["semantic_tokens"] for feature in features]

        acoustic_tokens = [feature["acoustic_tokens"] for feature in features]

        text_tokens = [torch.tensor(t) + self.num_special_tokens for t in text_byte_strings]

        semantic_tokens = [
            torch.tensor(t).squeeze(1) + self.num_special_tokens + self.num_text_tokens for t in semantic_tokens
        ]

        acoustic_tokens = [torch.tensor(t) for t in acoustic_tokens]

        min_speech_len = min(t.shape[0] for t in semantic_tokens)
        prompt_length = random.randint(min_speech_len // 8, min_speech_len // 2)

        random_prompt_starts = [random.randint(0, t.shape[0] - prompt_length) for t in acoustic_tokens]
        acoustic_prompts = [t[start:start + prompt_length] for t, start in zip(acoustic_tokens, random_prompt_starts)]
        acoustic_prompts = torch.stack(acoustic_prompts).transpose(1, 2)  # btq -> bqt

        joint_token_sequence = []
        for text, speech in zip(text_tokens, semantic_tokens):
            joint_sequence = torch.cat(
                [
                    torch.tensor([self.special_tokens["text"]]),
                    text,
                    torch.tensor([self.special_tokens["sep"]]),
                    torch.tensor([self.special_tokens["speech"]]),
                    speech,
                    torch.tensor([self.special_tokens["sep"]]),
                ]
            )
            joint_token_sequence.append(joint_sequence)

        # pad the joint token sequences to max length
        max_len = max(t.shape[-1] for t in joint_token_sequence)

        joint_token_sequence = [
            torch.nn.functional.pad(t, (0, max_len - len(t)), value=self.special_tokens["pad"]) for t in joint_token_sequence
        ]
        input_ids = torch.stack(joint_token_sequence)
        attention_mask = input_ids != self.special_tokens["pad"]

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "acoustic_prompts": acoustic_prompts,
        }

        return output


def filter_fn(examples):
    semantic_lengths = [len(semantic) for semantic in examples["semantic_tokens"]]
    text_lengths = [len(text) for text in examples["transcription_bytes"]]

    decisions = []
    for text_len, semantic_len in zip(text_lengths, semantic_lengths):
        decision = 1250 > semantic_len > text_len and semantic_len > 20
        decisions.append(decision)

    return decisions


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
    os.environ["WANDB_WATCH"] = "all"  # log gradients and model parameters

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
        print(f"Length of dataset before filtering: {len(train_dataset)}")
        train_dataset = train_dataset.filter(filter_fn, batched=True, num_proc=64, batch_size=64)
        print(f"Length of dataset after filtering: {len(train_dataset)}")
        train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if data_args.preprocessing_only:
        return

    # initialize random model
    if model_args.model_type == "text_to_semantic_w_length":
        config_class = TextToSemanticWLenConfig
        model_class = TextToSemanticWLen
    else:
        raise ValueError(f"Invalid model type: {model_args.model_type}")
    # load the base config
    config = config_class.from_pretrained(model_args.model_name_or_path)

    if model_args.extra_model_params is not None:
        config.update(model_args.extra_model_params)
    # reinitialize the config with updated dict (this is a hack)
    config = config_class(**config.to_dict())
    model = model_class(config).to(training_args.device)

    callbacks = [EndTrainingCallback(model_args.max_train_steps)]
    if data_args.time_limit is not None:
        callbacks.append(TimeLimitCallback(data_args.time_limit))

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(special_tokens=config.special_tokens),
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
