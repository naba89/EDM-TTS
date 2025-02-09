import math
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm
from transformers import PreTrainedModel, PretrainedConfig


def pad_signal_for_stft(
        signal: torch.Tensor, window_length: int, hop_length: int, match_stride: bool,
        pad_type: str = "reflect"
):
    """Compute how the STFT should be padded, based on match\_stride.

    Parameters
    ----------
    signal : torch.Tensor
        Length of audio signal.
    window_length : int
        Window length of STFT.
    hop_length : int
        Hop length of STFT.
    match_stride : bool
        Whether to match stride, making the STFT have the same alignment as
        convolutional layers.
    pad_type : str, optional

    Returns
    -------
    tuple
        Amount to pad on either side of audio.
    """
    length = signal.shape[-1]

    if match_stride:
        assert (
                hop_length == window_length // 4
        ), "For match_stride, hop must equal n_fft // 4"
        right_pad = math.ceil(length / hop_length) * hop_length - length
        pad = (window_length - hop_length) // 2
    else:
        right_pad = 0
        pad = 0

    signal = torch.nn.functional.pad(signal, (pad, right_pad), pad_type)

    return signal


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def convs(ch):
    return nn.ModuleList(
        [
            WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
        ]
    )


class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(1, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=sample_rate // rate
        )

    def forward(self, x):
        x = self.resample(x)

        fmap = []

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    def __init__(
            self,
            window_length: int,
            hop_factor: float = 0.25,
            sample_rate: int = 44100,
            bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=window_length,
            hop_length=int(window_length * hop_factor),
            power=None,
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32

        self.band_convs = nn.ModuleList([convs(ch) for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        x = pad_signal_for_stft(x, self.window_length,
                                int(self.window_length * self.hop_factor),
                                match_stride=True)
        x = self.spec_transform(x)[..., 2:-2]
        x = torch.view_as_real(x)
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")
        # Split into bands
        x_bands = [x[..., b[0]: b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class DACDiscriminatorConfig(PretrainedConfig):
    model_type = 'dac_discriminator'

    """
    Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rates: list = []
        self.periods: list = [2, 3, 5, 7, 11]
        self.fft_sizes: list = [2048, 1024, 512]
        self.sample_rate: int = 44100
        self.bands: list = BANDS
        self.__dict__.update(kwargs)


class DACDiscriminator(PreTrainedModel):
    config_class = DACDiscriminatorConfig
    def prepare_inputs_for_generation(self, *args, **kwargs):
        pass

    def _reorder_cache(self, past_key_values, beam_idx):
        pass

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        pass

    def __init__(self, config: DACDiscriminatorConfig):
        super().__init__(config)
        self.config = config
        discs = []
        discs += [MPD(p) for p in config.periods]
        discs += [MSD(r, sample_rate=config.sample_rate) for r in config.rates]
        discs += [MRD(f, sample_rate=config.sample_rate, bands=config.bands) for f in config.fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        fmaps = [d(x) for d in self.discriminators]
        return fmaps


if __name__ == "__main__":

    disc_config = DACDiscriminatorConfig()
    disc = DACDiscriminator(disc_config)

    inp = torch.zeros(1, 1, 44100)
    results = disc(inp)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for j, res in enumerate(result):
            print(res.shape, res.mean(), res.min(), res.max())
        print()
