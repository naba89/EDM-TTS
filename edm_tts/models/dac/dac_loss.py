import typing
from typing import List

import torch
import torch.nn.functional as F
import torchaudio.transforms
from torch import nn


class L1Loss(nn.L1Loss):
    """Weighted L1 Loss between two audio signals.

    Parameters
    ----------
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    """

    def __init__(self, weight: float = 1.0, **kwargs):
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Estimate AudioSignal
        y : torch.Tensor
            Reference AudioSignal

        Returns
        -------
        torch.Tensor
            L1 loss between the two signals
        """
        return super().forward(x, y)


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation adapted from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(
        self,
        scaling: int = True,
        reduction: str = "mean",
        zero_mean: int = True,
        clip_min: int = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        eps = 1e-8
        # nb, nc, nt
        references = x
        estimates = y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    power : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        weight: float = 1.0,
        power: float = 2.0,
    ):
        super().__init__()

        self.spec_transforms = nn.ModuleList([
            torchaudio.transforms.Spectrogram(
                n_fft=w,
                hop_length=w // 4,
                power=1.0,
            ) for w in window_lengths
        ])

        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = power

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for spec_transform in self.spec_transforms:
            x_mag = spec_transform(x)
            y_mag = spec_transform(y)
            loss += self.log_weight * self.loss_fn(
                x_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mag, y_mag)

        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        power: float = 2.0,
        weight: float = 1.0,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
    ):
        super().__init__()
        # create torchaudio mel transforms
        self.sample_rate = sample_rate
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=w,
                win_length=w,
                hop_length=w // 4,
                n_mels=n_mels,
                f_min=fmin,
                f_max=fmax,
            ) for w, n_mels, fmin, fmax in zip(window_lengths, n_mels, mel_fmin, mel_fmax)])

        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.pow = power

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0

        for mel_transform in self.mel_transforms:
            x_mels = mel_transform(x)
            y_mels = mel_transform(y)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self, sample_rate, waveform_args, multi_scale_stft_args, mel_spectrogram_args):
        super().__init__()
        self.waveform_loss = None if waveform_args is None else L1Loss(**waveform_args)
        self.multi_scale_stft_loss = None if multi_scale_stft_args is None else MultiScaleSTFTLoss(
            **multi_scale_stft_args)
        self.mel_spectrogram_loss = MelSpectrogramLoss(sample_rate=sample_rate, **mel_spectrogram_args)

    def forward(self, x, y):
        loss_dict = {}
        if self.waveform_loss is not None:
            loss_dict["waveform/loss"] = self.waveform_loss(x, y)
        if self.multi_scale_stft_loss is not None:
            loss_dict["stft/loss"] = self.multi_scale_stft_loss(x, y)
        loss_dict["mel/loss"] = self.mel_spectrogram_loss(x, y)
        # loss_dict = {
        #     "waveform/loss": self.waveform_loss(x, y),
        #     "stft/loss": self.multi_scale_stft_loss(x, y),
        #     "mel/loss": self.mel_spectrogram_loss(x, y)
        # }
        return loss_dict


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator=None):
        super().__init__()
        self.discriminator = discriminator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def forward(self, fake, real, loss_type):
        if loss_type == 'generator':
            return self.generator_loss(fake, real)
        elif loss_type == 'discriminator':
            return self.discriminator_loss(fake, real)

    def forward_discriminator(self, fake, real):
        if self.discriminator is None:
            raise ValueError("Discriminator not set! Call set_discriminator(discriminator) first.")
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        if self.discriminator is None:
            raise ValueError("Discriminator not set! Call set_discriminator(discriminator) first.")
        d_fake, d_real = self.forward_discriminator(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)

        return {'adv/disc_loss': loss_d}

    def generator_loss(self, fake, real):
        if self.discriminator is None:
            raise ValueError("Discriminator not set! Call set_discriminator(discriminator) first.")
        d_fake, d_real = self.forward_discriminator(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature
