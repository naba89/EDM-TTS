import functools
import os
import sys
from dataclasses import dataclass, field
from typing import List

import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from transformers import (
    set_seed, HfArgumentParser, TrainingArguments)

from edm_tts.datasets.audio_loading_utils import load_audio_segments, silence_filter, volume_normalize
from edm_tts.models.dac import DACConfig, DAC
from edm_tts.models.dac.dac_loss import GANLoss, ReconstructionLoss
from edm_tts.models.dac.discriminators import DACDiscriminatorConfig, DACDiscriminator
from edm_tts.trainers.gan_trainer import GANTrainer

from edm_tts.utils.utils import setup_logging

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    wandb_project: str = field(
        default="nar_speech",
        metadata={"help": "Wandb project name."},
    )

    generator_args: dict = field(
        default_factory=dict,
        metadata={"help": "Generator model arguments."},
    )

    discriminator_args: dict = field(
        default_factory=dict,
        metadata={"help": "Discriminator model arguments."},
    )

    gen_optimizer_name: str = field(
        default='Adam',
        metadata={"help": "The name of the generator optimizer to use."},
    )

    gen_optimizer_args: dict = field(
        default_factory=dict,
        metadata={"help": "Generator optimizer arguments."},
    )

    gen_scheduler_name: str = field(
        default='ExponentialLR',
        metadata={"help": "The name of the generator scheduler to use."},
    )

    gen_scheduler_args: dict = field(
        default_factory=dict,
        metadata={"help": "Generator scheduler arguments."},
    )

    disc_optimizer_name: str = field(
        default='Adam',
        metadata={"help": "The name of the discriminator optimizer to use."},
    )

    disc_optimizer_args: dict = field(
        default_factory=dict,
        metadata={"help": "Discriminator optimizer arguments."},
    )

    disc_scheduler_name: str = field(
        default='ExponentialLR',
        metadata={"help": "The name of the discriminator scheduler to use."},
    )

    disc_scheduler_args: dict = field(
        default=None,
        metadata={"help": "Discriminator scheduler arguments."},
    )

    waveform_args: dict = field(
        default=None,
        metadata={"help": "Waveform loss arguments."},
    )

    multi_scale_stft_args: dict = field(
        default_factory=dict,
        metadata={"help": "Multi-scale STFT loss arguments."},
    )

    mel_spectrogram_args: dict = field(
        default_factory=dict,
        metadata={"help": "Mel spectrogram loss arguments."},
    )

    lambdas: dict = field(
        default_factory=dict,
        metadata={"help": "Lambdas."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_args: dict = field(
        default_factory=dict,
        metadata={"help": "Dataset arguments."},
    )

    remove_columns: List[str] = field(
        default_factory=list,
        metadata={"help": "Columns to remove."}
    )

    validation_split: int = field(
        default=16,
        metadata={"help": "Validation split."},
    )

    num_shards: int = field(
        default=1024,
        metadata={"help": "Number of shards."},
    )

    shuffle_buffer_size: int = field(
        default=10000,
        metadata={"help": "Shuffle buffer size."},
    )

    silence_threshold: float = field(
        default=-40.0,
        metadata={"help": "Silence threshold."},
    )

    volume_normalize: float = field(
        default=-16.0,
        metadata={"help": "Volume normalize."},
    )

    volume_normalize_batch_size: int = field(
        default=16,
        metadata={"help": "Volume normalize batch size."},
    )

    preprocessing_only: bool = field(
        default=False,
        metadata={"help": "Preprocessing only."},
    )

    dataset_length: int = field(
        default=None,
        metadata={"help": "Dataset length."},
    )

    training_segment_length: float = field(
        default=0.38,
        metadata={"help": "Training segment length."},
    )

    validation_segment_length: float = field(
        default=5.0,
        metadata={"help": "Validation segment length."},
    )

    num_samples_to_log: int = field(
        default=5,
        metadata={"help": "Number of samples to log."},
    )


def main():

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

    # Setup logging
    setup_logging(logger, training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # intialize generator and discriminator
    generator_config = DACConfig(**model_args.generator_args)
    generator = DAC.from_config(generator_config).to(training_args.device)

    discriminator_config = DACDiscriminatorConfig(**model_args.discriminator_args)
    discriminator = DACDiscriminator.from_config(discriminator_config).to(training_args.device)

    with training_args.main_process_first(desc="dataset loading"):
        dataset = load_dataset(**data_args.dataset_args,
                               # world_size=training_args.world_size,
                               trust_remote_code=True)
        dataset_dict = dataset.train_test_split(test_size=data_args.validation_split)

        train_dataset = dataset_dict["train"]
        original_num_train_examples = len(train_dataset) * (
                    data_args.dataset_args['segment_length'] // data_args.training_segment_length)

        num_samples = int(data_args.dataset_length or original_num_train_examples)
        num_batches_per_epoch = int(num_samples //
                                    training_args.per_device_train_batch_size
                                    // training_args.world_size)

        train_dataset = train_dataset.to_iterable_dataset(num_shards=data_args.num_shards)

        train_dataset = train_dataset.shuffle(buffer_size=data_args.shuffle_buffer_size)

        train_dataset = train_dataset.map(functools.partial(load_audio_segments,
                                                            target_sr=generator.sample_rate,
                                                            segment_length=data_args.training_segment_length),
                                          batched=True, batch_size=1,
                                          remove_columns=data_args.remove_columns)

        if data_args.silence_threshold is not None:
            train_dataset = train_dataset.filter(
                functools.partial(silence_filter, threshold=data_args.silence_threshold),
                batched=True, batch_size=data_args.volume_normalize_batch_size)

        if data_args.volume_normalize is not None:
            train_dataset = train_dataset.map(functools.partial(volume_normalize, dbfs=data_args.volume_normalize),
                                              batched=True, batch_size=data_args.volume_normalize_batch_size)

        val_dataset = dataset_dict["test"]
        val_dataset.set_transform(functools.partial(load_audio_segments,
                                                    target_sr=generator.sample_rate,
                                                    segment_length=data_args.validation_segment_length))

    if data_args.preprocessing_only:
        return

    # initialize optimizer, scheduler, and loss function

    optimizerG = getattr(torch.optim, model_args.gen_optimizer_name)(generator.parameters(),
                                                                     **model_args.gen_optimizer_args)
    optimizerD = getattr(torch.optim, model_args.disc_optimizer_name)(discriminator.parameters(),
                                                                      **model_args.disc_optimizer_args)
    schedulerG = getattr(torch.optim.lr_scheduler, model_args.gen_scheduler_name)(optimizerG,
                                                                                  **model_args.gen_scheduler_args)
    schedulerD = getattr(torch.optim.lr_scheduler, model_args.disc_scheduler_name)(optimizerD,
                                                                                   **model_args.disc_scheduler_args)

    gan_loss = GANLoss()
    gan_loss.set_discriminator(discriminator)
    reconstruction_loss = ReconstructionLoss(
        sample_rate=generator.sample_rate,
        waveform_args=model_args.waveform_args,
        multi_scale_stft_args=model_args.multi_scale_stft_args,
        mel_spectrogram_args=model_args.mel_spectrogram_args)

    # Initialize Trainer
    trainer = GANTrainer(
        training_args=training_args,
        model_generator=generator,
        optimizer_generator=optimizerG,
        reconstruction_criterion=reconstruction_loss,
        scheduler_generator=schedulerG,
        model_discriminator=discriminator,
        optimizer_discriminator=optimizerD,
        gan_criterion=gan_loss,
        scheduler_discriminator=schedulerD,
        loss_lambdas=model_args.lambdas,
        train_dataset=train_dataset,
        collate_fn=None,
        num_batches_per_epoch=num_batches_per_epoch,
        num_samples_to_log=data_args.num_samples_to_log,
        sample_rate=generator.sample_rate,
        eval_dataset=val_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
