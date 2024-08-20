import warnings

import torch
import torch.nn as nn
import torchaudio
from audiotools import AudioSignal


def compute_output_lengths(input_lengths, kernel_size, stride, padding, dilation=1):
    return torch.floor((input_lengths + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1).int()


class AudioTokenizer(nn.Module):
    def __init__(self, acoustic_model, semantic_model):
        super().__init__()
        self.acoustic_model = acoustic_model
        self.semantic_model = semantic_model

    @property
    def sample_rate(self):
        assert self.acoustic_model.sample_rate == self.semantic_model.sample_rate
        return self.acoustic_model.sample_rate

    @property
    def device(self):
        assert self.acoustic_model.device == self.semantic_model.device
        return next(self.acoustic_model.parameters()).device

    @property
    def downsample_factor(self):
        # todo: double check
        return 320

    def pad(self, x):
        pad_val = (self.downsample_factor - x.shape[-1] % self.downsample_factor) % self.downsample_factor
        pad_left = pad_val // 2
        pad_right = pad_val - pad_left
        extra = self.downsample_factor // 4  # hack to ensure same length tokens
        return nn.functional.pad(x, (pad_left + extra, pad_right + extra))

    @torch.inference_mode()
    def volume_normalize(self, audio_batch, dbfs):
        signals = AudioSignal.batch([AudioSignal(audio, sample_rate=self.sample_rate) for audio in audio_batch])
        input_db = signals.loudness()
        signals = signals.normalize(db=dbfs)
        signals.ensure_max_of_audio()
        return signals.audio_data, input_db

    @torch.inference_mode()
    def compute_codes(self, audio_batch):
        audio_batch = self.pad(audio_batch)
        normalized_audio, input_db = self.volume_normalize(audio_batch.clone(), dbfs=-16)
        acoustic_codes = self.acoustic_model.encode_to_codes(normalized_audio)
        semantic_codes = self.semantic_model.encode(audio_batch)

        if acoustic_codes.shape[-1] != semantic_codes.shape[-1]:
            raise ValueError("Acoustic and semantic codes have different lengths")

        output = {
            "acoustic_codes": acoustic_codes,
            "semantic_codes": semantic_codes,
            "input_db": input_db,
        }
        return output

    @torch.inference_mode()
    def compute_codes_batch(self, acoustic_tokenizer_inputs, semantic_tokenizer_inputs):
        acoustic_codes = self.acoustic_model.encode_to_codes(**acoustic_tokenizer_inputs)
        semantic_codes = self.semantic_model.encode_batch(**semantic_tokenizer_inputs)

        if acoustic_codes.shape[-1] != semantic_codes.shape[-1]:
            raise ValueError("Acoustic and semantic codes have different lengths")

        output = {
            "acoustic_codes": acoustic_codes,
            "semantic_codes": semantic_codes,
        }
        return output

    def get_code_lengths(self, input_lengths):
        lengths = input_lengths
        for layer in self.acoustic_model.encoder.modules():
            if isinstance(layer, torch.nn.Conv1d):
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                padding = layer.padding[0]
                dilation = layer.dilation[0]
                lengths = compute_output_lengths(lengths, kernel_size, stride, padding, dilation)
        return lengths

    @torch.inference_mode()
    def compute_codes_from_file(self, file_path, offset=0, num_frames=-1, sample_rate=None):

        audio, sr = torchaudio.load(file_path, frame_offset=offset, num_frames=num_frames)
        if sample_rate is not None and sr != sample_rate:
            warnings.warn(f"Sample rate mismatch, between data and provided sample rate, {file_path}")

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio.unsqueeze(0)
        return self.compute_codes(audio)
