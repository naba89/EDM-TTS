# coding=utf-8

# Lint as: python3
"""Libri-Heavy audio dataset.
"""


import glob
import os

import math

import gzip
import json

import datasets


def normalize_text(s: str) -> str:
    s = s.replace("‘", "'")
    s = s.replace("’", "'")
    tokens = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'")
    s_list = [x.upper() if x in tokens else " " for x in s]
    s = " ".join("".join(s_list).split()).strip()
    return s


def clean_text(s: str) -> str:
    table = str.maketrans("’‘，。；？！（）：-《》、“”【】", "'',.;?!(): <>/\"\"[]")
    s = s.translate(table)
    return s.strip()


class LibriHeavyConfig(datasets.BuilderConfig):
    """BuilderConfig for Libri-Heavy."""

    def __init__(self, **kwargs):
        super(LibriHeavyConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


class LibriHeavy(datasets.GeneratorBasedBuilder):
    """Libri-Heavy dataset."""

    BUILDER_CONFIGS = [
        LibriHeavyConfig(name="small", description="577 hours, 35 GB."),
        LibriHeavyConfig(name="medium", description="5193 hours, 321 GB."),
        LibriHeavyConfig(name="large", description="51934 hours, 3.05 TB."),
        LibriHeavyConfig(name="all", description="60K hours, 3.4 TB."),
    ]

    @property
    def manual_download_instructions(self):
        return (
            "Currently, libriheavy should be downloaded and extracted. The extracted folder should contain "
            " the small, medium and large folders. Then you can specify the data_dir as the path to the folder "
            "`datasets.load_dataset('/path/to/libriheavy.py', data_dir='path/to/folder/folder_name')`"
        )

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "file": datasets.Value("string"),
                'sample_rate': datasets.Value('int64'),
                'offset': datasets.Value('int64'),
                'num_frames': datasets.Value('int64'),
                'transcription': datasets.Value('string'),
                'no_punc_transcription': datasets.Value('string'),
                'transcription_bytes': datasets.Sequence(datasets.Value('uint8')),
                'no_punc_transcription_bytes': datasets.Sequence(datasets.Value('uint8')),
            }
        )

        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        base_data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        self.base_data_dir = base_data_dir
        if not os.path.exists(base_data_dir):
            raise FileNotFoundError(
                f"{base_data_dir} does not exist. Make sure you insert a manual dir via "
                f"`datasets.load_dataset('/path/to/libriheavy.py', data_dir=...)` "
                f"that includes files unzipped from the librilight "
                f"dataset. Manual download instructions: {self.manual_download_instructions}"
            )
        small_files = glob.glob(os.path.join(base_data_dir, 'libriheavy', 'small', '*.jsonl.gz'))
        medium_files = glob.glob(os.path.join(base_data_dir, 'libriheavy', 'medium', '*.jsonl.gz'))
        large_files = glob.glob(os.path.join(base_data_dir, 'libriheavy', 'large', '*.jsonl.gz'))

        if self.config.name == "all":
            heavy_files = small_files + medium_files + large_files
        elif self.config.name == "small":
            heavy_files = small_files
        elif self.config.name == "medium":
            heavy_files = medium_files
        elif self.config.name == "large":
            heavy_files = large_files
        else:
            raise ValueError(f"Unknown config name {self.config.name}")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"heavy_files": heavy_files},
            ),
        ]

    def _generate_examples(self, heavy_files):
        for heavy_file in heavy_files:
            with gzip.open(heavy_file, 'rt', encoding='utf-8') as file:
                for line in file:
                    cut = json.loads(line)
                    seg_id = cut["id"]
                    start = math.floor(1000 * cut["start"]) / 1000
                    duration = math.floor(1000 * cut["duration"]) / 1000
                    wav_id = cut["recording"]["id"]
                    wav_path = os.path.join(self.base_data_dir, wav_id + ".flac")

                    text = cut["supervisions"][0]["custom"]["texts"][0]
                    no_punc_text = normalize_text(text)
                    punc_text = clean_text(text)

                    sample_rate = cut["recording"]["sampling_rate"]
                    start_samples = int(start * sample_rate)
                    duration_samples = int(duration * sample_rate)

                    punc_trans_bytes = list(punc_text.encode("utf-8"))
                    no_punc_trans_bytes = list(no_punc_text.encode("utf-8"))

                    if len(no_punc_trans_bytes) == 0 or len(punc_trans_bytes) == 0:
                        continue  # skip if utf-8 decoding fails to create a valid list of integers

                    yield seg_id, {
                        "id": seg_id,
                        "file": wav_path,
                        'sample_rate': sample_rate,
                        'offset': start_samples,
                        'num_frames': duration_samples,
                        'transcription': punc_text,
                        'no_punc_transcription': no_punc_text,
                        'transcription_bytes': punc_trans_bytes,
                        'no_punc_transcription_bytes': no_punc_trans_bytes,
                    }
