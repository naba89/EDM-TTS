# Lint as: python3
"""Libri-Light audio dataset.
"""
import glob
import os

import json

import datasets
import torchaudio


class LibriLightConfig(datasets.BuilderConfig):
    """BuilderConfig for Libri-Light."""

    def __init__(self, segment_length, **kwargs):
        super(LibriLightConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)
        self.segment_length = segment_length


class LibriLight(datasets.GeneratorBasedBuilder):
    """Libri-Light dataset."""

    BUILDER_CONFIGS = [
        LibriLightConfig(name="small", segment_length=60.0, description="577 hours, 35 GB."),
        LibriLightConfig(name="medium", segment_length=60.0, description="5193 hours, 321 GB."),
        LibriLightConfig(name="large", segment_length=60.0, description="51934 hours, 3.05 TB."),
        LibriLightConfig(name="all", segment_length=60.0, description="60K hours, 3.4 TB."),
    ]

    @property
    def manual_download_instructions(self):
        return (
            "Currently, librilight should be downloaded and extracted. The extracted folder should contain "
            " the small, medium and large folders. Then you can specify the data_dir as the path to the folder "
            "`datasets.load_dataset('/path/to/librilight.py', data_dir='path/to/folder/folder_name')`"
        )

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "file": datasets.Value("string"),
                'sample_rate': datasets.Value('int64'),
                'offset': datasets.Value('int64'),
                'num_frames': datasets.Value('int64'),
                'padding': datasets.Value('int64'),

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
                f"`datasets.load_dataset('matinf', data_dir=...)` that includes files unzipped from the librilight "
                f"dataset. Manual download instructions: {self.manual_download_instructions}"
            )
        if self.config.name == "all":
            data_dirs = glob.glob(os.path.join(base_data_dir, "small", "**", "**", "*.flac")) + \
                        glob.glob(os.path.join(base_data_dir, "medium", "**", "**", "*.flac")) + \
                        glob.glob(os.path.join(base_data_dir, "large", "**", "**", "*.flac"))
        else:
            data_dirs = glob.glob(os.path.join(base_data_dir, self.config.name, "**", "**", "*.flac"))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dirs": data_dirs},
            ),
        ]

    def _generate_examples(self, data_dirs):
        for key, path in enumerate(data_dirs):
            path_split = path.split("/")
            id_ = '/'.join(path_split[-4:]).replace(".flac", "")

            audio_metadata = torchaudio.info(path)
            segment_length = int(self.config.segment_length * audio_metadata.sample_rate)
            total_length = audio_metadata.num_frames

            # generate non-overlapping segments of segment_length
            # formatted as (offset, num_frames, padding) in samples
            segments = []
            start = 0
            while start < total_length:
                end = start + segment_length - 1
                padding = 0
                if end > total_length:
                    end = total_length
                    padding = segment_length - (end - start)
                segments.append((start, end, padding))
                start = end + 1

            for segment_id, (start, end, padding) in enumerate(segments):
                _id_ = f'{id_}_{segment_id}'
                yield _id_, {
                    "id": _id_,
                    "file": path,
                    'sample_rate': audio_metadata.sample_rate,
                    'offset': start,
                    'num_frames': end-start+1,
                    'padding': padding,
                }
