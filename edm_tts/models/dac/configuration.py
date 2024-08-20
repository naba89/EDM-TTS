from typing import List, Union

from transformers import PretrainedConfig


class DACConfig(PretrainedConfig):
    model_type = "dac"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_dim: int = 64
        self.encoder_rates: List[int] = [2, 4, 5, 8]
        self.decoder_dim: int = 1536
        self.decoder_rates: List[int] = [8, 5, 4, 2]
        self.n_codebooks: int = 12
        self.codebook_size: int = 1024
        self.codebook_dim: Union[int, list] = 8
        self.quantizer_dropout: float = 0.5
        self.sample_rate: int = 16000
        self.__dict__.update(kwargs)


if __name__ == '__main__':
    cfg = DACConfig()
    cfg.save_pretrained("../../../configs/dac/base_config")
