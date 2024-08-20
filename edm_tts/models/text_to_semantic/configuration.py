from transformers import PretrainedConfig


class TextToSemanticWLenConfig(PretrainedConfig):
    model_type = "text_to_semantic_w_length"

    def __init__(
            self,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,

            hidden_size=512,
            semantic_vocab_size=1024,  # 1024 semantic tokens
            text_vocab_size=256,  # 256 byte tokens

            attn_flash=True,

            main_encoder_num_heads=16,
            main_encoder_num_layers=8,
            main_encoder_ff_mult=4,
            main_encoder_conv_kernel_size=5,
            main_encoder_attn_dropout=0.0,
            main_encoder_ff_dropout=0.0,
            main_encoder_conv_dropout=0.0,

            length_predictor_num_heads=16,
            length_predictor_num_layers=4,
            length_predictor_ff_mult=4,
            length_predictor_conv_kernel_size=5,
            length_predictor_attn_dropout=0.0,
            length_predictor_ff_dropout=0.0,
            length_predictor_conv_dropout=0.0,

            main_encoder_args=None,
            length_predictor_args=None,
            **kwargs
    ):
        super().__init__(
            **kwargs,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        self.hidden_size = hidden_size
        self.text_vocab_size = text_vocab_size
        self.semantic_vocab_size = semantic_vocab_size

        self.special_tokens = {
            "pad": 0,
            "text": 1,
            "speech": 2,
            "sep": 3,
            "mask": 4,
        }

        self.main_encoder_args = {
            "depth": main_encoder_num_layers,
            "heads": main_encoder_num_heads,
            "ff_mult": main_encoder_ff_mult,
            "conv_kernel_size": main_encoder_conv_kernel_size,
            "attn_dropout": main_encoder_attn_dropout,
            "ff_dropout": main_encoder_ff_dropout,
            "conv_dropout": main_encoder_conv_dropout,
            "attn_flash": attn_flash,
        }

        self.length_predictor_args = {
            "depth": length_predictor_num_layers,
            "heads": length_predictor_num_heads,
            "ff_mult": length_predictor_ff_mult,
            "conv_kernel_size": length_predictor_conv_kernel_size,
            "attn_dropout": length_predictor_attn_dropout,
            "ff_dropout": length_predictor_ff_dropout,
            "conv_dropout": length_predictor_conv_dropout,
            "attn_flash": attn_flash,
        }

        if main_encoder_args is not None:
            self.main_encoder_args.update(main_encoder_args)

        if length_predictor_args is not None:
            self.length_predictor_args.update(length_predictor_args)

        self.length_predictor_args["dim_head"] = hidden_size // length_predictor_num_heads
        self.main_encoder_args["dim_head"] = hidden_size // main_encoder_num_heads


if __name__ == '__main__':
    config = TextToSemanticWLenConfig()
    config.save_pretrained("../../../configs/text_to_semantic_w_length/base_config")
