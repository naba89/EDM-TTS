# Lint as: python3
"""semantic and acoustic codes dataset with text.
"""


import glob
import os

import datasets
import torch


class TextSpeechCodesDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for Text-SpeechCodes dataset."""

    def __init__(self, **kwargs):
        super(TextSpeechCodesDatasetConfig, self).__init__(**kwargs)


class TextSpeechCodesDataset(datasets.GeneratorBasedBuilder):
    """Codes dataset."""

    BUILDER_CONFIGS = [
        TextSpeechCodesDatasetConfig(name="all", description="TextSpeechCodes dataset"),
    ]

    @property
    def manual_download_instructions(self):
        return (
            "Codes should be computed before using this dataset. "
            "`datasets.load_dataset('/path/to/this/script', name=all, data_dir='path/to/folder/folder_name/of/codes')`"
        )

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "length": datasets.Value("int32"),
                "transcription": datasets.Value("string"),
                "acoustic_tokens": datasets.Array2D(shape=(None, 12), dtype="int16"),
                "semantic_tokens": datasets.Array2D(shape=(None, 1), dtype="int16"),
                "transcription_bytes": datasets.Sequence(datasets.Value("uint8")),
            }
        )

        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        base_data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        if not os.path.exists(base_data_dir):
            raise FileNotFoundError(
                f"{base_data_dir} does not exist. Make sure you insert a manual dir via "
                f"`datasets.load_dataset('/this/script', data_dir=...)` "
                f"that includes code files .pt files "
                f"dataset. Manual download instructions: {self.manual_download_instructions}"
            )

        train_data_dirs = glob.glob(os.path.join(base_data_dir, "**", "*.pt"), recursive=True)
        print(f"Found {len(train_data_dirs)} files")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dirs": train_data_dirs},
            ),
        ]

    def _generate_examples(self, data_dirs):
        for key, path in enumerate(data_dirs):
            id_ = path.split("/")[-1].replace(".pt", "")

            data = torch.load(path, map_location="cpu")
            for i, (k, v) in enumerate(data.items()):
                acoustic_tokens = v["acoustic_codes"]
                semantic_tokens = v["semantic_codes"]

                if acoustic_tokens.ndim == 3:
                    acoustic_tokens = acoustic_tokens.squeeze(0).transpose(0, 1)
                else:
                    acoustic_tokens = acoustic_tokens.transpose(0, 1)
                if semantic_tokens.ndim == 2:
                    semantic_tokens = semantic_tokens.transpose(0, 1)
                else:
                    semantic_tokens = semantic_tokens.unsqueeze(1)

                transcription = v["transcription"]
                transcription_bytes = list(transcription.encode("utf-8"))

                yield f"{id_}_{i}", {
                    "id": f"{id_}_{i}",
                    "length": semantic_tokens.shape[0] + len(transcription_bytes),
                    "transcription": transcription,
                    "transcription_bytes": transcription_bytes,
                    "acoustic_tokens": acoustic_tokens,
                    "semantic_tokens": semantic_tokens,
                }
