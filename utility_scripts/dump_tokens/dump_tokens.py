import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from edm_tts.models.audio_tokenizer.audio_tokenizer import AudioTokenizer
from transformers import AutoFeatureExtractor

from edm_tts.models.audio_tokenizer.semantic_tokenizer_hubert import SemanticModelHuBERT
from edm_tts.models.dac import DAC


def prepare_dataset(dataset_args):
    streaming = dataset_args.get("streaming", False)
    num_proc = 64 if not streaming else None
    return load_dataset(**dataset_args, num_proc=num_proc, trust_remote_code=True)


def get_dataset_args(dataset_name, split):
    if dataset_name == "librilight":
        return {
            "path": "edm_tts/datasets/librilight.py",
            "name": "full",
            "split": split,
            "data_dir": "data/libri-light/unlab",
            "streaming": False,
            "cache_dir": "cache/librilight",
        }
    elif dataset_name == "libriheavy-small":  # 122526
        return {
            "path": "edm_tts/datasets/libriheavy.py",
            "name": "small",
            "split": split,
            "data_dir": "data/libri-light/unlab",
            "streaming": False,
            "cache_dir": "cache/libriheavy",
        }
    elif dataset_name == "libriheavy-medium":  # 1101040
        return {
            "path": "edm_tts/datasets/libriheavy.py",
            "name": "medium",
            "split": split,
            "data_dir": "data/libri-light/unlab",
            "streaming": False,
            "cache_dir": "cache/libriheavy",
        }
    elif dataset_name == "libriheavy-large":  # 11156939
        return {
            "path": "edm_tts/datasets/libriheavy.py",
            "name": "large",
            "split": split,
            "data_dir": "data/libri-light/unlab",
            "streaming": False,
            "cache_dir": "cache/libriheavy",
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def pad(x, downsample_factor):
    pad_val = (downsample_factor - x.shape[-1] % downsample_factor) % downsample_factor
    pad_left = pad_val // 2
    pad_right = pad_val - pad_left
    extra = downsample_factor // 4  # hack to ensure same length tokens
    return torch.nn.functional.pad(x, (pad_left + extra, pad_right + extra))


@dataclass
class Collator:
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")
    downsample_factor = 320

    def __call__(self, batch):
        filenames = [b["file"] for b in batch]
        transcriptions = [b["transcription"] for b in batch] if "transcription" in batch[0] else None
        no_punc_transcriptions = [b["no_punc_transcription"] for b in batch] if "no_punc_transcription" in batch[0] else None
        transcription_bytes = [b["transcription_bytes"] for b in batch] if "transcription_bytes" in batch[0] else None
        no_punc_transcription_bytes = [b["no_punc_transcription_bytes"] for b in batch] if "no_punc_transcription_bytes" in batch[0] else None
        ids = [b["id"] for b in batch]
        offsets = [b["offset"] for b in batch]
        num_frames = [b["num_frames"] for b in batch]
        paddings = [b["padding"] for b in batch] if "padding" in batch[0] else [0] * len(batch)

        audios = []
        audios_np = []
        audio_lengths = []
        for filename, offset, num_frame, padding in zip(filenames, offsets, num_frames, paddings):
            audio, sr = torchaudio.load(filename, frame_offset=offset, num_frames=num_frame)
            audio = F.pad(audio, (0, padding))
            audio = pad(audio, self.downsample_factor).mean(0)
            audios.append(audio)
            audios_np.append(audio.numpy())
            audio_lengths.append(audio.shape[-1])

        audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True).unsqueeze(1)
        semantic_tokenizer_inputs = self.feature_extractor(audios_np, return_tensors="pt",
                                                           sampling_rate=16000, padding="longest",
                                                           return_attention_mask=True)
        return {
            "semantic_tokenizer_inputs": {"input_values": semantic_tokenizer_inputs["input_values"],
                                          "attention_mask": semantic_tokenizer_inputs["attention_mask"]},
            "acoustic_tokenizer_inputs": {"audio": audios_padded},
            "transcriptions": transcriptions,
            "no_punc_transcriptions": no_punc_transcriptions,
            "transcription_bytes": transcription_bytes,
            "no_punc_transcription_bytes": no_punc_transcription_bytes,
            "ids": ids,
            "audio_lengths": torch.tensor(audio_lengths),
        }


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset_name", type=str, default="libriheavy-small")
    arg_parser.add_argument("--split", type=str, default="train")
    arg_parser.add_argument("--codec_model", type=str, required=True)
    arg_parser.add_argument("--semantic_centroid_path", type=str, default=None)
    arg_parser.add_argument("--semantic_model", type=str, default="facebook/hubert-large-ll60k")
    arg_parser.add_argument("--semantic_layer_idx", type=int, default=18)

    arg_parser.add_argument("--output_dir", type=str, required=True)
    arg_parser.add_argument("--max_files_per_output_file", type=int, default=1000)

    args = arg_parser.parse_args()

    accelerator = Accelerator(dispatch_batches=False)
    device = accelerator.device
    print(f"Using device: {device}")

    dataset_args = get_dataset_args(args.dataset_name, args.split)

    # Load the dataset
    with accelerator.main_process_first():
        dataset = prepare_dataset(dataset_args)

    accelerator.print("Overall dataset length", len(dataset))

    # Initialize model and feature extractor

    semantic_model = SemanticModelHuBERT(
        model_name=args.semantic_model,
        cluster_centers_path=args.semantic_centroid_path,
        output_layer=args.semantic_layer_idx,
    ).eval()
    acoustic_model = DAC.from_pretrained(args.codec_model).eval()

    audio_tokenizer = AudioTokenizer(acoustic_model=acoustic_model, semantic_model=semantic_model).eval().to(device)

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=Collator(),
                            num_workers=32)

    dataloader = accelerator.prepare(dataloader)

    accelerator.print("Prepared dataloader length", len(dataloader))

    output_dir = os.path.join(args.output_dir, f"{args.dataset_name}_{args.split}")
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    max_files_per_output_file = args.max_files_per_output_file
    out_aggregate = {}
    rank = accelerator.process_index
    fname_index = 0

    counter = 0
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        save_fname = os.path.join(output_dir, f"{rank}_{fname_index}.pt")

        transcriptions = batch["transcriptions"]
        no_punc_transcriptions = batch["no_punc_transcriptions"]
        transcription_bytes = batch["transcription_bytes"]
        no_punc_transcription_bytes = batch["no_punc_transcription_bytes"]
        ids = batch["ids"]

        audio_lengths = batch["audio_lengths"]
        code_lengths = audio_tokenizer.get_code_lengths(audio_lengths)

        acoustic_tokenizer_inputs = batch["acoustic_tokenizer_inputs"]
        semantic_tokenizer_inputs = batch["semantic_tokenizer_inputs"]

        for key, value in acoustic_tokenizer_inputs.items():
            acoustic_tokenizer_inputs[key] = value.to(device)
        for key, value in semantic_tokenizer_inputs.items():
            semantic_tokenizer_inputs[key] = value.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.inference_mode():
            tokens = audio_tokenizer.compute_codes_batch(acoustic_tokenizer_inputs=acoustic_tokenizer_inputs,
                                                         semantic_tokenizer_inputs=semantic_tokenizer_inputs)

        for i, name in enumerate(ids):

            tokens_i = {
                "acoustic_codes": tokens["acoustic_codes"][i, :, :code_lengths[i]],
                "semantic_codes": tokens["semantic_codes"][i, :code_lengths[i]],
            }

            if transcriptions is not None:
                tokens_i.update({
                    "transcription": transcriptions[i],
                })
            if no_punc_transcriptions is not None:
                tokens_i.update({
                    "no_punc_transcription": no_punc_transcriptions[i],
                })
            if transcription_bytes is not None:
                tokens_i.update({
                    "transcription_bytes": transcription_bytes[i],
                })
            if no_punc_transcription_bytes is not None:
                tokens_i.update({
                    "no_punc_transcription_bytes": no_punc_transcription_bytes[i],
                })

            out_aggregate[name] = tokens_i

        counter += batch_size
        if counter >= max_files_per_output_file:
            torch.save(out_aggregate, save_fname)
            out_aggregate = {}
            fname_index += 1
            counter = 0

    if out_aggregate:
        torch.save(out_aggregate, os.path.join(output_dir, f"{rank}_{fname_index}.pt"))

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
