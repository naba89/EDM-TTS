from typing import Sequence

import torch

from edm_tts.utils.text.symbols import SYMBOLS, SYMBOL_TO_ID, ID_TO_SYMBOL


class LetterTokenizer:
    def __init__(self) -> None:
        self.num_text_tokens = len(SYMBOLS)
        self.word_delimiter_token = " "

    def encode(
            self,
            sequences: str | Sequence[str],
            num_special_token: int = 0,
            label_maxlen: int | None = None,
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
        if label_maxlen:
            return torch.tensor([SYMBOL_TO_ID[symbol] for symbol in text][:label_maxlen], dtype=torch.long,
                                device=device) + num_special_token
        else:
            return torch.tensor([SYMBOL_TO_ID[symbol] for symbol in text],
                                dtype=torch.long, device=device) + num_special_token

    def decode(self, sequences: list[torch.Tensor], num_special_token: int = 0) -> list[str]:
        """Decodes given label sequences to text."""
        return [self._decode_single((sequence - num_special_token).tolist()) for sequence in sequences]

    def _decode_single(self, labels: list[int]) -> str:
        """convert indices of token id back into text"""

        text = "".join([ID_TO_SYMBOL[label] for label in labels])
        text = text.replace(self.word_delimiter_token, " ")
        return text
