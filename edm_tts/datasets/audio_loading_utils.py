import random

import numpy as np
import torch
import torchaudio
from audiotools import AudioSignal


def load_code_segments(examples, segment_length=None, random_segment=False,
                       acoustic_sample_rate=16000,
                       acoustic_downsample_factor=4,
                       semantic_sample_rate=16000, semantic_downsample_factor=4):

    acoustic_tokens = torch.as_tensor(np.stack(examples['acoustic_tokens'], axis=0)).long().transpose(1, 2)
    semantic_tokens = torch.as_tensor(np.stack(examples['semantic_tokens'], axis=0)).squeeze(-1).long()

    # above will anyways fail if the tokens are not of the same length
    assert acoustic_tokens.shape[0] == 1, "For now, batch size must be 1"

    if segment_length is None:
        acoustic_token_length = acoustic_tokens.shape[-1]
        semantic_token_length = semantic_tokens.shape[-1]
    else:
        acoustic_token_length = int(segment_length * acoustic_sample_rate / acoustic_downsample_factor)
        semantic_token_length = int(segment_length * semantic_sample_rate / semantic_downsample_factor)

    if acoustic_tokens.shape[-1] < acoustic_token_length or semantic_tokens.shape[-1] < semantic_token_length:
        # drop the example if the tokens are too short
        output = {
            'acoustic_tokens': [],
            'semantic_tokens': []
        }
        return output

    # select random start point
    batch_size = acoustic_tokens.shape[0]
    assert batch_size == 1, "For now, batch size must be 1"
    if random_segment:
        acoustic_start = random.randint(0, acoustic_tokens.shape[-1] - acoustic_token_length)
        # map to semantic start point
        semantic_start = acoustic_start * acoustic_downsample_factor // semantic_downsample_factor
    else:
        acoustic_start = 0
        semantic_start = 0

    # select the tokens
    acoustic_tokens = acoustic_tokens[:, :, acoustic_start:acoustic_start + acoustic_token_length]
    semantic_tokens = semantic_tokens[:, semantic_start:semantic_start + semantic_token_length]

    acoustic_tokens = torch.unbind(acoustic_tokens, dim=0)
    semantic_tokens = torch.unbind(semantic_tokens, dim=0)

    output = {
        'acoustic_tokens': acoustic_tokens,
        'semantic_tokens': semantic_tokens
    }

    return output


def load_audio_segments(examples, target_sr, segment_length=None):
    def _load(fname, start, num_frames, padding):
        audio, sample_rate = torchaudio.load(fname, frame_offset=start, num_frames=num_frames)
        if padding > 0:
            audio = torch.nn.functional.pad(audio, (0, padding))

        if sample_rate != target_sr:
            audio = torchaudio.transforms.Resample(sample_rate, target_sr)(audio)
        return audio

    audios = [_load(f, st, nf, p) for f, st, nf, p in zip(examples['file'],
                                                          examples['offset'],
                                                          examples['num_frames'],
                                                          examples['padding'])]
    out_audios = []
    ids = []
    sample_rates = []
    for i, aud in enumerate(audios):
        if segment_length is None:
            segment_length_samples = aud.shape[1]
        else:
            segment_length_samples = int(segment_length * target_sr)
        splits = torch.split(aud, segment_length_samples, dim=1)
        # drop last split if it is less than segment_length_samples
        if splits[-1].shape[1] < segment_length_samples:
            splits = splits[:-1]
        out_audios += splits
        ids += [f"{examples['id'][i]}-{j}" for j in range(len(splits))]
        sample_rates += [target_sr] * len(splits)

    output = {
        'id': ids,
        'audio': out_audios,
        'sample_rate': sample_rates,
    }
    return output


def silence_filter(examples, threshold=-40):
    audios = examples['audio']
    srs = examples['sample_rate']
    signals = AudioSignal.batch([AudioSignal(audio, sample_rate=sr) for audio, sr in zip(audios, srs)])
    loudness = signals.loudness()
    return (loudness > threshold).tolist()


def volume_normalize(examples, dbfs):
    audios = examples['audio']
    srs = examples['sample_rate']
    signals = AudioSignal.batch([AudioSignal(audio, sample_rate=sr) for audio, sr in zip(audios, srs)])
    signals = signals.normalize(db=dbfs)
    return {'audio': [signal.audio_data[0] for signal in signals]}
