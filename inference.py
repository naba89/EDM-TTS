import argparse

import torch
import torchaudio

from edm_tts.models.audio_tokenizer.audio_tokenizer import AudioTokenizer
from edm_tts.models.audio_tokenizer.semantic_tokenizer_hubert import SemanticModelHuBERT
from edm_tts.models.dac import DAC
from edm_tts.models.text_to_semantic.modeling_text_to_semantic import TextToSemanticWLen
from edm_tts.models.injection_conformer.modeling_injection_conformer import InjectionConformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--speaker_prompt", type=str, required=True)
    argparser.add_argument("-t", "--text", type=str, required=True)
    argparser.add_argument("-o", "--output", type=str, required=True)
    # argparser.add_argument("--codec_model", type=str, default="exp/edm_tts/dac/best_model")
    # argparser.add_argument("--t2s_model", type=str, default="exp/edm_tts/text_to_semantic_w_length/")
    # argparser.add_argument("--s2a_model", type=str, default="exp/edm_tts/injection_conformer/")

    argparser.add_argument("--codec_model", type=str, default="subatomicseer/acoustic_tokenizer")
    argparser.add_argument("--t2s_model", type=str, default='/data/umeiro0/users/nabarun/projects/github/aaai/exp/aaai/text_to_semantic/libriheavy_text_to_semantic_w_length/checkpoint-300000')
    argparser.add_argument("--s2a_model", type=str, default="/data/umeiro0/users/nabarun/projects/github/aaai/exp/aaai/lightning_speech_hubert_dac_ll60k/checkpoint-100000")

    args = argparser.parse_args()

    semantic_model = SemanticModelHuBERT()
    acoustic_model = DAC.from_pretrained(args.codec_model)
    audio_tokenizer = AudioTokenizer(acoustic_model=acoustic_model, semantic_model=semantic_model).eval().to(device)

    s2a_model = InjectionConformerModel.from_pretrained(args.s2a_model).eval().to(device)
    t2s_model = TextToSemanticWLen.from_pretrained(args.t2s_model).eval().to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.inference_mode():
        prompt_tokens = audio_tokenizer.compute_codes_from_file(args.speaker_prompt)
        prompt_semantic_tokens = prompt_tokens["semantic_codes"][0].long().to(device)
        prompt_acoustic_tokens = prompt_tokens["acoustic_codes"][0].long().to(device)

        out_semantic_tokens = t2s_model.infer(text=args.text,
                                              pred_iters=16,
                                              temperature=1.0,
                                              ).speech_pred_tokens[None]

        out_acoustic_tokens = s2a_model.infer_special(
            semantic_tokens=out_semantic_tokens,
            acoustic_prompt_tokens=prompt_acoustic_tokens[None],
            semantic_prompt_tokens=prompt_semantic_tokens[None],
            steps=8,
        )
        out_audio = audio_tokenizer.acoustic_model.decode_from_codes(out_acoustic_tokens)[0].float().cpu()

    torchaudio.save(args.output, out_audio, 16000)


if __name__ == '__main__':
    main()
