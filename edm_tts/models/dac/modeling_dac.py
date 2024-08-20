import math

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional
from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin

from .configuration import DACConfig
from .decoder import Decoder
from .encoder import Encoder
from .vector_quantizer import ResidualVectorQuantize


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)  # type: ignore


class DAC(PreTrainedModel, ModuleUtilsMixin):
    config_class = DACConfig

    def resize_position_embeddings(self, new_num_position_embeddings: int): ...

    def get_position_embeddings(self) -> nn.Embedding | tuple[nn.Embedding]: ...

    def prepare_inputs_for_generation(self, *args, **kwargs): ...

    def _reorder_cache(self, past_key_values, beam_idx): ...

    def __init__(
        self,
        config: DACConfig,
    ):
        super().__init__(config)

        self.encoder_dim = config.encoder_dim
        self.encoder_rates = config.encoder_rates
        self.decoder_dim = config.decoder_dim
        self.decoder_rates = config.decoder_rates
        self.sample_rate = config.sample_rate

        self.hop_length = np.prod(config.decoder_rates)
        self.encoder = Encoder(config.encoder_dim, config.encoder_rates)

        self.n_codebooks = config.n_codebooks
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim

        self.quantizer = ResidualVectorQuantize(
            self.encoder.enc_dim,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
        )
        self.latent_dim = self.encoder.enc_dim

        self.decoder = Decoder(
            self.encoder.enc_dim,
            config.decoder_dim,
            config.decoder_rates,
        )
        self.sample_rate = config.sample_rate
        self.downsample_factor = np.prod(config.decoder_rates)

        self.apply(init_weights)

    @property
    def device(self):
        return next(self.parameters()).device

    def preprocess(self, audio_data: torch.Tensor, sample_rate: int | None) -> tuple[torch.Tensor, int]:
        """perform resampling for audio if

        Args:
            audio_data (torch.Tensor): tensor of audio
            sample_rate (int | None): original sampling_ratio of the audio

        Returns:
            tuple[torch.Tensor, int]: resampled audio tensor and its length
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        else:
            audio_data = torchaudio.functional.resample(audio_data, sample_rate, self.sample_rate)

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, int(right_pad)))
        return audio_data, length

    def encode(
        self,
        audio_data: torch.Tensor,
        sample_rate: int | None = None,
        n_quantizers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        out = {}
        audio_data, length = self.preprocess(audio_data, sample_rate)
        out["length"] = length

        out["z"] = self.encoder(audio_data)
        out.update(self.quantizer(out["z"], n_quantizers))
        return out

    def decode(self, z: torch.Tensor, length: int | None = None) -> dict[str, torch.Tensor]:
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        out = {}
        x = self.decoder(z)
        out["audio"] = x[..., :length]
        return out

    def encode_to_codes(self, audio, n_quantizers=None) -> torch.Tensor:
        audio = audio.to(self.device)
        z = self.encoder(audio)
        codes = self.quantizer(z, n_quantizers)["codes"]
        return codes

    def decode_from_codes(self, codes: torch.Tensor, length: int | None = None) -> torch.Tensor:
        z = self.quantizer.from_codes(codes)[0]
        return self.decode(z, length)["audio"]

    def codes_to_features(self, codes: torch.Tensor):
        z = self.quantizer.from_codes(codes)[0]
        return z

    @torch.no_grad()
    def features_to_codes(self, features: torch.Tensor):
        return self.quantizer.from_latents(features)[-1]

    def codes_to_features_unreduced(self, codes: torch.Tensor):
        return self.quantizer.from_codes_unreduced(codes)

    def features_to_codebook_logits(self, features: torch.Tensor):  # bdt
        return self.quantizer.latents_to_codebook_dist(features)  # btqn

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int | None = None,
        n_quantizers: int | None = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        out = {}
        out.update(self.encode(audio_data, sample_rate, n_quantizers))
        out.update(self.decode(out["z"], out["length"]))
        return out

    def remove_weight_norm(self):
        for module in self.modules():
            if hasattr(module, "weight_g"):
                nn.utils.remove_weight_norm(module)
