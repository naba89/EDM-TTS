
# Lint as: python3
"""semantic and acoustic codes dataset.
"""


import glob
import os

import datasets
import torch
import numpy as np


class CodesDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for Codes."""

    def __init__(self, **kwargs):
        super(CodesDatasetConfig, self).__init__(**kwargs)


class CodesDataset(datasets.GeneratorBasedBuilder):
    """Codes dataset."""

    BUILDER_CONFIGS = [
        CodesDatasetConfig(name="all", description="Code dataset"),
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
                "acoustic_tokens": datasets.Array2D(shape=(None, 12), dtype="int16"),
                "semantic_tokens": datasets.Array2D(shape=(None, 1), dtype="int16"),
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
                f"that includes code files .npy files "
                f"dataset. Manual download instructions: {self.manual_download_instructions}"
            )

        train_data_dirs = glob.glob(os.path.join(base_data_dir, "**", "*.pt"), recursive=True)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dirs": train_data_dirs},
            ),
        ]

    def _generate_examples(self, data_dirs):
        for key, path in enumerate(data_dirs):
            id_ = path.split("/")[-1].replace(".pt", "")

            data = torch.load(path, map_location='cpu')
            for i, (k, v) in enumerate(data.items()):
                acoustic_tokens = v["acoustic_codes"].transpose(0, 1)
                semantic_tokens = v["semantic_codes"][..., None]

                yield f"{id_}_{i}", {
                    "id": f"{id_}_{i}",
                    "length": acoustic_tokens.shape[0],
                    "acoustic_tokens": acoustic_tokens,
                    "semantic_tokens": semantic_tokens,
                }
