from transformers import PretrainedConfig


class SpeechTextMultiTaskConfig(PretrainedConfig):
    model_type = "speech_text_multitask"

    def __init__(
        self,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_size: int = 384,
        num_text_tokens: int = 256,
        num_speech_tokens: int = 1024,
        use_speech_mlm: bool = True,
        use_ctc_correction: bool = True,
        encoder_num_layers: int = 10,
        encoder_num_heads: int = 8,
        encoder_ff_mult: int = 4,
        encoder_conv_kernel_size: int = 15,
        encoder_conv_expansion_factor: int = 2,
        encoder_attn_dropout: float = 0.1,
        encoder_ff_dropout: float = 0.1,
        encoder_conv_dropout: float = 0.1,

        tokenizer_name: str = "letter",
        speech_tokenizer_name: str | None = None,
        augment: bool = False,

        attn_flash: bool = True,
        encoder_config: dict | None = None,
        **kwargs
    ):
        super().__init__(
            **kwargs,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        self.num_text_tokens = num_text_tokens
        self.num_speech_tokens = num_speech_tokens
        self.tokenizer_name = tokenizer_name
        self.speech_tokenizer_name = speech_tokenizer_name
        self.hidden_size = hidden_size

        self.augment = augment

        self.use_speech_mlm = use_speech_mlm
        self.use_ctc_correction = use_ctc_correction

        self.encoder_config = {
            "depth": encoder_num_layers,
            "heads": encoder_num_heads,
            "ff_mult": encoder_ff_mult,
            "conv_expansion_factor": encoder_conv_expansion_factor,
            "conv_kernel_size": encoder_conv_kernel_size,
            "attn_dropout": encoder_attn_dropout,
            "ff_dropout": encoder_ff_dropout,
            "conv_dropout": encoder_conv_dropout,
            "attn_flash": attn_flash,
        }

        if encoder_config is not None:
            self.encoder_config.update(encoder_config)

        self.encoder_config["dim_head"] = hidden_size // encoder_num_heads


if __name__ == '__main__':
    config = SpeechTextMultiTaskConfig()
    config.save_pretrained("../../../../configs/speech_text/speech_text_multitask/base_config")

    config = SpeechTextMultiTaskConfig.from_pretrained(
        "../../../../configs/speech_text/speech_text_multitask/base_config"
    )
