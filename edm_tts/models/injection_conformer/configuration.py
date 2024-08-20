from transformers import PretrainedConfig


class InjectionConformerConfig(PretrainedConfig):
    model_type = "injection_conformer"

    def __init__(
            self,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,

            hidden_size=1024,
            num_semantic_tokens=1024,

            acoustic_model_path="exp/edm_tts/dac/best_model",

            attn_flash=True,

            encoder_num_heads=16,
            encoder_num_layers=16,
            encoder_ff_mult=4,
            encoder_conv_kernel_size=5,
            encoder_attn_dropout=0.1,
            encoder_ff_dropout=0.1,
            encoder_conv_dropout=0.1,

            injection_layers=(4, 7, 10, 13),
            residual=True,
            use_injection=True,
            loss_all=False,

            encoder_config=None,
            **kwargs
    ):
        super().__init__(
            **kwargs,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        self.hidden_size = hidden_size
        self.num_semantic_tokens = num_semantic_tokens
        self.acoustic_model_path = acoustic_model_path

        self.encoder_config = {
                "depth": encoder_num_layers,
                "heads": encoder_num_heads,
                "ff_mult": encoder_ff_mult,
                "conv_kernel_size": encoder_conv_kernel_size,
                "attn_dropout": encoder_attn_dropout,
                "ff_dropout": encoder_ff_dropout,
                "conv_dropout": encoder_conv_dropout,
                "attn_flash": attn_flash,
            }
        self.residual = residual
        self.use_injection = use_injection
        self.loss_all = loss_all

        if encoder_config is not None:
            self.encoder_config.update(encoder_config)

        self.encoder_config["dim_head"] = hidden_size // encoder_num_heads

        self.injection_layers = injection_layers


if __name__ == '__main__':
    config = InjectionConformerConfig()
    config.save_pretrained("../../../configs/injection_conformer/base_config")
