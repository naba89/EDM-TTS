import os

import torch
from tqdm.auto import tqdm

import wandb
import accelerate
from accelerate import DistributedDataParallelKwargs, Accelerator
from torch.utils.data import DataLoader


class CheckpointMetadata:
    """Extra metadata for training."""

    current_epoch = 0
    overall_step = 0
    best_val_loss = float("inf")

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class GANTrainer:
    def __init__(self,
                 training_args,
                 model_generator,
                 optimizer_generator,
                 reconstruction_criterion,
                 scheduler_generator,
                 model_discriminator,
                 optimizer_discriminator,
                 gan_criterion,
                 scheduler_discriminator,
                 loss_lambdas,
                 train_dataset,
                 collate_fn,
                 num_batches_per_epoch,
                 num_samples_to_log,
                 sample_rate,
                 eval_dataset,
                 ):
        self.training_args = training_args
        self.output_dir = training_args.output_dir

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
        self.accelerator = Accelerator(log_with=training_args.report_to, kwargs_handlers=[kwargs])
        self.device = self.accelerator.device
        self.run_dir = training_args.output_dir

        if self.accelerator.is_main_process:
            config_dict = training_args.to_dict()
            logger_kwargs = {"wandb": {"name": training_args.output_dir}}
            project_name = os.getenv("WANDB_PROJECT", "huggingface")
            self.accelerator.init_trackers(project_name, config_dict, init_kwargs=logger_kwargs)
            os.makedirs(self.run_dir, exist_ok=True)

        # set the seed
        accelerate.utils.set_seed(training_args.seed)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        dataloader_params = {
            "batch_size": training_args.per_device_train_batch_size,
            "collate_fn": collate_fn,
            "num_workers": training_args.dataloader_num_workers,
            "pin_memory": training_args.dataloader_pin_memory,
            "persistent_workers": training_args.dataloader_persistent_workers,
        }

        self.train_loader = DataLoader(self.train_dataset, **dataloader_params)
        dataloader_params["batch_size"] = training_args.per_device_eval_batch_size
        self.eval_loader = DataLoader(self.eval_dataset, **dataloader_params)

        self.num_batches_per_epoch = num_batches_per_epoch

        stuff_to_prepare = [
            model_generator,
            optimizer_generator,
            reconstruction_criterion,
            scheduler_generator,
            model_discriminator,
            optimizer_discriminator,
            gan_criterion,
            scheduler_discriminator,
            self.train_loader,
            self.eval_loader
        ]

        self.generator, self.optimizerG, self.reconstruction_loss, self.schedulerG, \
            self.discriminator, self.optimizerD, self.gan_loss, self.schedulerD, \
            self.train_loader, self.eval_loader = self.accelerator.prepare(*stuff_to_prepare)

        self.lambdas = loss_lambdas

        self.total_epochs = training_args.num_train_epochs
        self.max_steps = training_args.max_steps

        self.train_metadata = CheckpointMetadata()

        self.accelerator.register_for_checkpointing(self.train_metadata)

        self.samples_to_log = None

        self.num_samples_to_log = num_samples_to_log
        self.sample_rate = sample_rate

    def train(self):
        self._train()

    def train_step(self, batch):
        self.generator.train()
        self.discriminator.train()
        output = {}

        audio = batch['audio'].to(self.device)
        sample_rate = batch['sample_rate'][0].to(self.device)

        out = self.generator(audio, sample_rate)

        recons = out["audio"]

        # get the extra loss terms returned by model, if any, e.g. vq commitment loss and codebook loss
        for k, v in out.items():
            if 'loss' in k:
                output[k] = v

        disc_loss = self.gan_loss(recons, audio, loss_type="discriminator")
        output.update(disc_loss)
        self.optimizerD.zero_grad(set_to_none=True)
        self.accelerator.backward(output["adv/disc_loss"])
        self.optimizerD.step()
        self.schedulerD.step()

        output.update(self.reconstruction_loss(recons, audio))

        (
            output["adv/gen_loss"],
            output["adv/feat_loss"],
        ) = self.gan_loss(recons, audio, loss_type="generator")

        output["loss"] = sum([v * output[k] for k, v in self.lambdas.items() if k in output])

        self.optimizerG.zero_grad(set_to_none=True)
        self.accelerator.backward(output["loss"])
        self.optimizerG.step()
        self.schedulerG.step()

        if hasattr(self.accelerator.unwrap_model(self.generator), "set_num_updates"):
            self.accelerator.unwrap_model(self.generator).set_num_updates(self.train_metadata.overall_step)

        for k, v in output.items():
            output[k] = v.detach()

        if hasattr(out, 'vq/perplexity'):
            output['vq/perplexity'] = out['vq/perplexity'].detach()

        return output

    def _train(self):
        start_epoch = self.train_metadata.current_epoch
        for epoch in tqdm(range(start_epoch, self.total_epochs),
                          disable=not self.accelerator.is_local_main_process,
                          desc="Epochs", leave=True):
            if hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch)

            train_losses_dict = None

            for batch in tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process,
                              leave=False, total=self.num_batches_per_epoch,
                              desc="Batches"):
                loss_dict = self.train_step(batch)
                if train_losses_dict is None:
                    train_losses_dict = loss_dict
                else:
                    for k, v in loss_dict.items():
                        train_losses_dict[k] += v

                if (self.train_metadata.overall_step + 1) % self.training_args.save_steps == 0:
                    self.save_state()

                if (self.train_metadata.overall_step + 1) % self.training_args.eval_steps == 0:
                    val_loss = self._evaluate()
                    self.accelerator.log(
                        {
                            "val_loss": val_loss,
                        },
                        step=self.train_metadata.overall_step,
                    )
                    if val_loss < self.train_metadata.best_val_loss:
                        self.train_metadata.best_val_loss = val_loss
                        self.save_best()
                if (self.train_metadata.overall_step + 1) % self.training_args.logging_steps == 0:
                    for k, v in train_losses_dict.items():
                        train_losses_dict[k] = (v / self.training_args.logging_steps).item()
                    if 'vq/perplexities' in train_losses_dict:
                        perplexity = train_losses_dict.pop('vq/perplexities')
                        for i, p in enumerate(perplexity):
                            train_losses_dict[f'vq/perplexity_{i}'] = p.item()
                    self.accelerator.log(
                        train_losses_dict,
                        step=self.train_metadata.overall_step,
                    )
                    train_losses_dict = None
                if (self.train_metadata.overall_step + 1) % self.training_args.eval_steps == 0:
                    self.save_samples()

                self.train_metadata.overall_step += 1

                if self.train_metadata.overall_step % self.num_batches_per_epoch == 0:
                    break

            self.train_metadata.current_epoch += 1

        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Training finished.")
        self.accelerator.end_training()

    def evaluate(self, best=True, path=None):
        self.load_pretrained(best, path)
        self._evaluate()

    @torch.inference_mode()
    def eval_step(self, batch):
        audio = batch['audio'].to(self.device)
        sample_rate = batch['sample_rate'][0].to(self.device)
        recons = self.generator(audio, sample_rate)["audio"]
        recons, audio = self.accelerator.gather_for_metrics((recons, audio))
        self.samples_to_log = recons.detach()
        val_loss = self.reconstruction_loss.mel_spectrogram_loss(recons, audio)
        return val_loss.detach()

    def _evaluate(self):
        self.generator.eval()

        total_val_loss = 0
        for batch in self.eval_loader:
            loss = self.eval_step(batch)
            total_val_loss += loss

        val_loss = total_val_loss / len(self.eval_loader)
        return val_loss

    @torch.no_grad()
    def save_samples(self):

        if self.samples_to_log is None:
            return

        # samples = []
        for i in range(self.num_samples_to_log):
            self.accelerator.log({
                f"Audio Sample {i}": wandb.Audio(
                    self.samples_to_log[i].detach().cpu().t().numpy(),
                    caption=str(i),
                    sample_rate=self.sample_rate)},
                step=self.train_metadata.overall_step)

        self.samples_to_log = None

    def load_state(self, checkpoint_path):
        self.accelerator.load_state(checkpoint_path)

    def save_state(self):
        checkpoint_path = os.path.join(self.run_dir, f"checkpoint_{self.train_metadata.overall_step}")
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(checkpoint_path)

    def unwrapped_generator(self):
        return self.accelerator.unwrap_model(self.generator)

    def save_best(self):
        best_path = os.path.join(self.run_dir, "best_model")
        self.unwrapped_generator().save_pretrained(best_path, is_main_process=self.accelerator.is_main_process)

    def load_pretrained(self, best=True, path=None):
        if path is None and best:
            path = os.path.join(self.run_dir, "best_model")
        elif path is None and not best:
            raise ValueError("Either best=True or path should be provided.")
        self.unwrapped_generator().from_pretrained(path)
