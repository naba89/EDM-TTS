from typing import Sequence

import torch

from edm_tts.tokenizers.protocols import HuggingFaceTokenizerProtocol


class HuggingFaceTokenizer:
    def __init__(
        self,
        hugging_face_tokenizer: HuggingFaceTokenizerProtocol,
        num_text_tokens: int,
        sos_position: int | None = None,
        eos_position: int | None = None,
    ):
        """
        Hugging Face tokenizer wrapper

        Args:
            hugging_face_tokenizer: hugging face tokenizer
            num_text_tokens: will be used for model initialization, it must equal to actual num_text_tokens in tokenizer
            sos_position: Position where sentense starts, e.g. BertTokenizer will include unrelated token, like [CLS]
            eos_position: Position where sentense starts, e.g. BertTokenizer will include unrelated token, like [SEP]

        Raises:
            ValueError: when input num_text_tokens is different from num tokens in tokenizer
        """
        self.hugging_face_tokenizer = hugging_face_tokenizer
        self.sos_position = sos_position
        self.eos_position = eos_position
        expect_num_text_token = len(hugging_face_tokenizer.get_vocab())
        if num_text_tokens != expect_num_text_token:
            # help user to identify the excat num_text_tokens
            raise ValueError(
                f"You must set correct num_text_tokens for {hugging_face_tokenizer.name_or_path} in yaml config, current value is {num_text_tokens}, expected to be: {expect_num_text_token}."
            )
        self.num_text_tokens = num_text_tokens

    def encode(
        self, sequences: str | Sequence[str], num_special_token: int = 0, label_maxlen: int | None = None,
            device: torch.device = torch.device("cpu")
    ) -> list[torch.Tensor]:
        """Encodes given text sequences to label token sequences."""
        if isinstance(sequences, str):
            return [
                self._encode_single(sequences, num_special_token, label_maxlen, device),
            ]
        else:
            return [self._encode_single(seq, num_special_token, label_maxlen, device) for seq in sequences]

    def _encode_single(self, text: str, num_special_token: int, label_maxlen: int | None,
                       device: torch.device = torch.device("cpu")
                       ) -> torch.Tensor:
        """convert single string text"""
        encoded_tokens = self.hugging_face_tokenizer.encode(text)
        if self.sos_position:
            encoded_tokens = encoded_tokens[self.sos_position :]
        if self.eos_position:
            encoded_tokens = encoded_tokens[: self.eos_position]
        if label_maxlen:
            return torch.tensor(encoded_tokens[:label_maxlen], dtype=torch.long, device=device) + num_special_token
        else:
            return torch.tensor(encoded_tokens, dtype=torch.long, device=device) + num_special_token

    def decode(self, sequences: list[torch.Tensor], num_special_token: int = 0) -> list[str]:
        """Decodes given label sequences to text."""
        return [self._decode_single((sequence - num_special_token).tolist()) for sequence in sequences]

    def _decode_single(self, labels: list[int]) -> str:
        """convert indices of token id back into text"""
        return self.hugging_face_tokenizer.decode(labels)
