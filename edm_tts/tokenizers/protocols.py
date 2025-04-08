from typing import Protocol, Sequence

import torch


class TextTokenizerProtocol(Protocol):
    @property
    def num_text_tokens(self) -> int: ...

    def encode(
        self, sequences: Sequence[str], label_maxlen: int | None = None, num_special_token: int = 0
    ) -> list[torch.Tensor]: ...

    def decode(self, sequences: list[torch.Tensor], num_special_token: int = 0) -> list[str]: ...


class HuggingFaceTokenizerProtocol(Protocol):
    @property
    def name_or_path(self) -> str: ...

    def encode(self, line: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...

    def get_vocab(self) -> dict[int, str]: ...
