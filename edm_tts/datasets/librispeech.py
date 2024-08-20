import glob
import os
import warnings
from pathlib import Path

import datasets


class LibriSpeech(datasets.GeneratorBasedBuilder):
    """LibriSpeech dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="clean", description="Clean subset dataset"),
        datasets.BuilderConfig(name="other", description="Other subset dataset"),
        datasets.BuilderConfig(name="full", description="Full dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "file": datasets.Value("string"),
                    "transcription": datasets.Value("string"),
                }
            ),
        )

    def _load_transcripts(self, path):
        transcripts_glob = os.path.join(path, "**/*.trans.txt")
        transcripts = {}
        for transcript_path in glob.glob(transcripts_glob, recursive=True):
            if transcript_path.startswith((".", "__")):
                continue
            with open(transcript_path, mode="r", encoding="utf-8") as file:
                for line in file:
                    if line:
                        line = line.strip()
                        filename, transcript = line.split(" ", 1)
                        transcripts[filename] = transcript
        return transcripts

    def _split_generators(self, dl_manager):
        root_path = dl_manager.manual_dir or ""
        if not os.path.exists(root_path):
            raise FileNotFoundError(
                f"{root_path} does not exist. Make sure you insert a manual dir in the data_dir field. "
            )

        train_clean_100_dir = os.path.join(root_path, "LibriSpeech", "train-clean-100")
        train_clean_360_dir = os.path.join(root_path, "LibriSpeech", "train-clean-360")
        train_other_500_dir = os.path.join(root_path, "LibriSpeech", "train-other-500")

        dev_clean_dir = os.path.join(root_path, "LibriSpeech", "dev-clean")
        dev_other_dir = os.path.join(root_path, "LibriSpeech", "dev-other")

        test_clean_dir = os.path.join(root_path, "LibriSpeech", "test-clean")
        test_other_dir = os.path.join(root_path, "LibriSpeech", "test-other")

        transcripts_clean_100 = self._load_transcripts(train_clean_100_dir)
        transcripts_clean_360 = self._load_transcripts(train_clean_360_dir)
        transcripts_other_500 = self._load_transcripts(train_other_500_dir)

        transcripts_dev_clean = self._load_transcripts(dev_clean_dir)
        transcripts_dev_other = self._load_transcripts(dev_other_dir)

        transcripts_test_clean = self._load_transcripts(test_clean_dir)
        transcripts_test_other = self._load_transcripts(test_other_dir)

        audio_files_clean_100 = glob.glob(os.path.join(train_clean_100_dir, "**", "*.flac"), recursive=True)
        audio_files_clean_360 = glob.glob(os.path.join(train_clean_360_dir, "**", "*.flac"), recursive=True)
        audio_files_other_500 = glob.glob(os.path.join(train_other_500_dir, "**", "*.flac"), recursive=True)

        audio_files_dev_clean = glob.glob(os.path.join(dev_clean_dir, "**", "*.flac"), recursive=True)
        audio_files_dev_other = glob.glob(os.path.join(dev_other_dir, "**", "*.flac"), recursive=True)

        audio_files_test_clean = glob.glob(os.path.join(test_clean_dir, "**", "*.flac"), recursive=True)
        audio_files_test_other = glob.glob(os.path.join(test_other_dir, "**", "*.flac"), recursive=True)

        if self.config.name == "full":
            train_splits = [
                datasets.SplitGenerator(
                    name="train",
                    gen_kwargs={
                        "audio_files": audio_files_clean_100 + audio_files_clean_360 + audio_files_other_500,
                        "transcripts": transcripts_clean_100 | transcripts_clean_360 | transcripts_other_500,
                    },
                ),
            ]
            val_splits = [
                datasets.SplitGenerator(
                    name="dev",
                    gen_kwargs={
                        "audio_files": audio_files_dev_clean + audio_files_dev_other,
                        "transcripts": transcripts_dev_clean | transcripts_dev_other,
                    },
                ),
            ]
            test_splits = [
                datasets.SplitGenerator(
                    name="test",
                    gen_kwargs={
                        "audio_files": audio_files_test_clean + audio_files_test_other,
                        "transcripts": transcripts_test_clean | transcripts_test_other,
                    },
                ),
            ]
        elif self.config.name == "clean":
            train_splits = [
                datasets.SplitGenerator(
                    name="train",
                    gen_kwargs={
                        "audio_files": audio_files_clean_100 + audio_files_clean_360,
                        "transcripts": transcripts_clean_100 | transcripts_clean_360,
                    },
                ),
                datasets.SplitGenerator(
                    name="train_clean_100",
                    gen_kwargs={
                        "audio_files": audio_files_clean_100,
                        "transcripts": transcripts_clean_100,
                    },
                ),
                datasets.SplitGenerator(
                    name="train_clean_360",
                    gen_kwargs={
                        "audio_files": audio_files_clean_360,
                        "transcripts": transcripts_clean_360,
                    },
                ),
            ]
            val_splits = [
                datasets.SplitGenerator(
                    name="dev",
                    gen_kwargs={
                        "audio_files": audio_files_dev_clean,
                        "transcripts": transcripts_dev_clean,
                    },
                ),
            ]
            test_splits = [
                datasets.SplitGenerator(
                    name="test",
                    gen_kwargs={
                        "audio_files": audio_files_test_clean,
                        "transcripts": transcripts_test_clean,
                    },
                ),
            ]
        elif self.config.name == "other":
            train_splits = [
                datasets.SplitGenerator(
                    name="train",
                    gen_kwargs={
                        "audio_files": audio_files_other_500,
                        "transcripts": transcripts_other_500,
                    },
                ),
            ]
            val_splits = [
                datasets.SplitGenerator(
                    name="dev",
                    gen_kwargs={
                        "audio_files": audio_files_dev_other,
                        "transcripts": transcripts_dev_other,
                    },
                ),
            ]
            test_splits = [
                datasets.SplitGenerator(
                    name="test",
                    gen_kwargs={
                        "audio_files": audio_files_test_other,
                        "transcripts": transcripts_test_other,
                    },
                ),
            ]
        else:
            raise ValueError(f"Unknown dataset: {self.config.name}")

        return train_splits + val_splits + test_splits

    def _generate_examples(self, audio_files, transcripts):
        """Generate examples."""

        for audio_file in audio_files:
            # Skip files that do not end with '.flac' or start with '.' or '__'
            if not audio_file.endswith(".flac") or audio_file.startswith((".", "__")):
                continue
            uid = Path(audio_file).stem
            transcript = transcripts.get(uid, None)
            if transcript is None:
                warnings.warn(f"Transcript not found for file {audio_file}")
                continue

            example = {
                "id": uid,
                "file": audio_file,
                "transcription": transcript,
            }
            yield uid, example
