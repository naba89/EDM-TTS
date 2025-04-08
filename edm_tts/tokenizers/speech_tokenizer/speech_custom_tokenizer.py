from typing import Sequence

import torch
from transformers import AutoTokenizer


def map_to_chinese_char(token):
    return chr(0x4E00 + token)


# Map Chinese characters back to integer tokens
def map_to_integer(char):
    unicode_val = ord(char)
    if 0x4E00 <= unicode_val <= 0x9FFF:  # Only map if it's in the Chinese character range
        return unicode_val - 0x4E00
    else:
        return None  # Ignore characters that are not in the expected range


class SpeechCustomTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path: str | None = None,
        num_speech_tokens: int | None = None,
    ):
        """
        Hugging Face tokenizer wrapper

        Args:
            tokenizer_name_or_path: hugging face tokenizer
            num_speech_tokens: will be used for model initialization, it must equal to actual num_speech_tokens in tokenizer

        Raises:
            ValueError: when input num_text_tokens is different from num tokens in tokenizer
        """
        self.hugging_face_tokenizer = None
        self.num_speech_tokens = num_speech_tokens
        if tokenizer_name_or_path is not None:
            self.hugging_face_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            self.num_speech_tokens = self.hugging_face_tokenizer.vocab_size

        assert self.num_speech_tokens is not None, "num_speech_tokens must be provided"

    def encode(
        self, sequences: list[list[int]],
            device: torch.device = torch.device("cpu")
    ) -> list[torch.Tensor]:
        """Encodes given text sequences to label token sequences."""
        return [self._encode_single(seq, device) for seq in sequences]

    def _encode_single(self, sequence: list[int],
                       device: torch.device = torch.device("cpu")
                       ) -> torch.Tensor:
        """convert single string text"""
        if self.hugging_face_tokenizer is None:
            return torch.tensor(sequence, dtype=torch.long, device=device)
        sequence = ''.join([map_to_chinese_char(token) for token in sequence])
        encoded_tokens = self.hugging_face_tokenizer.encode(sequence)
        return torch.tensor(encoded_tokens, dtype=torch.long, device=device)

    def decode(self, sequences: list[torch.Tensor]) -> list[list[int]]:
        """Decodes given label sequences to text."""
        return [self._decode_single(sequence.tolist()) for sequence in sequences]

    def _decode_single(self, labels: list[int]) -> list[int]:
        """convert indices of token id back into text"""
        if self.hugging_face_tokenizer is None:
            return labels
        decoded = self.hugging_face_tokenizer.decode(labels)
        decoded = [map_to_integer(char) for char in decoded]
        decoded = [char for char in decoded if char is not None]
        return decoded
