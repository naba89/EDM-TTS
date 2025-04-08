import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.utils import ModelOutput
from monotonic_align import mask_from_lens
from super_monotonic_align import maximum_path

from edm_tts.models.common.discrete_augmentation.augmentation import time_warp
from edm_tts.tokenizers.speech_tokenizer.speech_custom_tokenizer import SpeechCustomTokenizer
from edm_tts.tokenizers.text_tokenizer.letter_tokenizer import LetterTokenizer
from edm_tts.models.conformer.conformer import Conformer
from edm_tts.utils.utils import random_topk_mask
from edm_tts.models.t2v2.configuration_t2v2 import SpeechTextMultiTaskConfig
from edm_tts.tokenizers import ByteTokenizer, HuggingFaceTokenizer


from edm_tts.utils.text.cleaners import english_cleaners
from edm_tts.utils.utils import ctc_decode



@dataclass
class SpeechTextMultiTaskModelOutput(ModelOutput):
    loss: torch.Tensor | None = None
    ctc_text_loss: torch.Tensor | None = None
    length_loss: torch.Tensor | None = None
    speech_mlm_loss: torch.Tensor | None = None
    align_speech_loss: torch.Tensor | None = None
    ctc_correction_loss: torch.Tensor | None = None


class SpeechTextMultiTask(PreTrainedModel, ModuleUtilsMixin):
    config_class = SpeechTextMultiTaskConfig

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        ...

    def get_position_embeddings(self) -> nn.Embedding | Tuple[nn.Embedding]:
        ...

    def prepare_inputs_for_generation(self, *args, **kwargs):
        ...

    def _reorder_cache(self, past_key_values, beam_idx):
        ...

    def __init__(
            self,
            config: SpeechTextMultiTaskConfig,
    ) -> None:
        super().__init__(config)

        self.config = config
        tokenizer_name = config.to_dict().get("tokenizer_name", "letter")
        if tokenizer_name == "byte":
            self.tokenizer = ByteTokenizer()
            self.num_text_tokens = self.tokenizer.num_text_tokens
        elif tokenizer_name == "letter":
            self.tokenizer = LetterTokenizer()
            self.num_text_tokens = self.tokenizer.num_text_tokens
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            self.tokenizer = HuggingFaceTokenizer(
                hugging_face_tokenizer=tokenizer,
                num_text_tokens=tokenizer.vocab_size,
                sos_position=1,
                eos_position=-1,
            )
            self.num_text_tokens = self.tokenizer.num_text_tokens

        self.register_buffer("false", torch.tensor([0], dtype=torch.bool))
        self.speech_tokenizer = SpeechCustomTokenizer(tokenizer_name_or_path=config.speech_tokenizer_name,
                                                      num_speech_tokens=config.num_speech_tokens)

        self.dim = config.hidden_size

        self.num_special_tokens = 0

        self.text_ctc_blank_token = 0  # CTC blank token, don't change this
        self.text_fill_token = self.num_text_tokens
        self.text_mask_token = self.num_text_tokens + 1
        self.text_pad_id = self.num_text_tokens + 2

        self.num_speech_tokens = self.speech_tokenizer.num_speech_tokens
        self.speech_pad_id = self.num_speech_tokens
        self.speech_mask_token = self.num_speech_tokens + 1

        self.total_speech_input_tokens = self.num_speech_tokens + 2  # +2 for pad and mask token
        self.total_text_input_tokens = self.num_text_tokens + 3  # +3 for pad and mask token and filler token

        self.speech_embedding = nn.Embedding(self.total_speech_input_tokens,
                                             self.dim, padding_idx=self.speech_pad_id)
        self.text_embedding = nn.Embedding(self.total_text_input_tokens,
                                           self.dim, padding_idx=self.text_pad_id)

        self.conformer = Conformer(dim=self.dim, **config.encoder_config)

        self.use_speech_mlm = config.use_speech_mlm
        self.use_ctc_correction = config.use_ctc_correction

        self.speech_pred_head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_speech_tokens, bias=True),
        )

        if self.use_ctc_correction:
            self.ctc_correction_pred_head = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.GELU(approximate="tanh"),
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.num_text_tokens + 1, bias=True),  # +1 for CTC-blank token
            )

        self.length_prediction_head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 1, bias=True),
        )

        self.ctc_text_pred_head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_text_tokens + 1, bias=True),  # +1 for CTC-blank token
        )

        self.speech_ce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.speech_pad_id)
        self.ctc_loss_fn = nn.CTCLoss(blank=self.text_ctc_blank_token, zero_infinity=True)
        if self.use_ctc_correction:
            self.text_ce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.text_pad_id)
        self.length_loss_fn = nn.L1Loss()

    def mas(self, ctc_log_probs, text_tokens, speech_attn_mask, text_attn_mask):
        """
        Masked alignment score
        :param ctc_log_probs: torch.Tensor, shape: [batch_size, seq_len, num_text_tokens + 1]
        :param speech_attn_mask: torch.Tensor, shape: [batch_size, seq_len]
        :param text_tokens: torch.Tensor, shape: [batch_size, seq_len]
        :param text_attn_mask: torch.Tensor, shape: [batch_size, seq_len]
        :return:
        """
        B, T, V = ctc_log_probs.size()
        text_lengths = text_attn_mask.sum(dim=-1).long()
        speech_lengths = speech_attn_mask.sum(dim=-1).long()
        adjusted_tokens = text_tokens.clone()
        adjusted_tokens[~text_attn_mask] = self.text_ctc_blank_token
        similarity = torch.gather(ctc_log_probs, 2,
                                  adjusted_tokens.unsqueeze(1).expand(-1, T, -1)).transpose(1, 2)  # (B, S, T)

        mask_ST = mask_from_lens(similarity, text_lengths, speech_lengths)
        alignment = maximum_path(similarity, mask_ST)

        return alignment

    def forward(self, speech_tokens, speech_attn_mask, text_tokens, text_attn_mask,
                ):
        """
        Forward pass of the model
        :param speech_tokens:  torch.Tensor, shape: [batch_size, seq_len]
        :param speech_attn_mask:   torch.Tensor, shape: [batch_size, seq_len]
        :param text_tokens:   torch.Tensor, shape: [batch_size, seq_len]
        :param text_attn_mask:  torch.Tensor, shape: [batch_size, seq_len]
        :return:
        """

        if self.config.augment:
            # time-warp augmentation (https://arxiv.org/pdf/2309.07377)
            speech_tokens = time_warp(speech_tokens, attn_mask=speech_attn_mask)

        batch_size = text_tokens.size(0)
        speech_lengths = speech_attn_mask.sum(dim=-1).long()
        text_lengths = text_attn_mask.sum(dim=-1).long()

        # Prepare inputs for the CTC task and MLM task
        speech_embeddings = self.speech_embedding(speech_tokens)
        speech_tokens_list = [speech_tokens[i, :speech_lengths[i]] for i in range(batch_size)]
        speech_masked_tokens, speech_masks, _ = self.cosine_schedule_masking(speech_tokens_list, uniform=False,
                                                                             mask_token=self.speech_mask_token)
        speech_masked_tokens = pad_sequence(speech_masked_tokens, batch_first=True, padding_value=self.speech_pad_id)
        speech_masks = pad_sequence(speech_masks, batch_first=True, padding_value=False)
        speech_masked_embeddings = self.speech_embedding(speech_masked_tokens)
        speech_mlm_targets = speech_tokens[speech_masks]

        # join batch-wise and run conformer and then split
        if self.use_speech_mlm:
            ctc_and_mlm_input_embeddings = torch.cat([speech_embeddings, speech_masked_embeddings], dim=0)
            ctc_and_mlm_attn_mask = torch.cat([speech_attn_mask, speech_attn_mask], dim=0)
        else:
            ctc_and_mlm_input_embeddings = speech_embeddings
            ctc_and_mlm_attn_mask = speech_attn_mask

        ctc_and_mlm_out, *_ = self.conformer(ctc_and_mlm_input_embeddings, mask=ctc_and_mlm_attn_mask)

        if self.use_speech_mlm:
            # MLM loss calculation
            mlm_out = ctc_and_mlm_out[speech_embeddings.size(0):]
            speech_mlm_logits = self.speech_pred_head(mlm_out[speech_masks])
            speech_mlm_loss = self.speech_ce_loss_fn(speech_mlm_logits.view(-1, speech_mlm_logits.size(-1)),
                                                     speech_mlm_targets.view(-1))
        else:
            speech_mlm_loss = torch.tensor(0.0, device=self.device)

        # CTC loss calculation
        ctc_out = ctc_and_mlm_out[:speech_embeddings.size(0)]
        ctc_logits = self.ctc_text_pred_head(ctc_out)
        ctc_targets = text_tokens + 1  # +1 for CTC-blank token
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1, dtype=torch.float32)
        ctc_loss = self.ctc_loss_fn(ctc_log_probs.transpose(0, 1), ctc_targets, speech_lengths, text_lengths)

        # Get alignment for Length prediction and Align TTS tasks
        text_embeddings = self.text_embedding(text_tokens)
        alignment = self.mas(ctc_log_probs.detach(), ctc_targets,
                             speech_attn_mask, text_attn_mask).to(text_embeddings.dtype)  # (B, T, S)
        lengths = alignment.sum(dim=-1)

        # Length prediction task
        length_out, *_ = self.conformer(text_embeddings, mask=text_attn_mask)
        length_preds = self.length_prediction_head(length_out[text_attn_mask]).squeeze(-1)
        length_targets = lengths[text_attn_mask].float().log()
        length_loss = self.length_loss_fn(length_preds, length_targets)

        # Prepare inputs for the Align TTS task and CTC-Correction task
        ctc_correction_input_embeddings = None
        if self.use_ctc_correction:
            # CTC-Correction task inputs
            ctc_raw_output = ctc_logits.detach().argmax(dim=-1)
            blank_mask = ctc_raw_output == self.text_ctc_blank_token
            ctc_raw_output = ctc_raw_output - 1
            ctc_raw_output = ctc_raw_output.masked_fill(blank_mask, self.text_fill_token)
            ctc_raw_output_list = [ctc_raw_output[i, :speech_lengths[i]] for i in range(batch_size)]
            masked_ctc_raw_output, _, _ = self.cosine_schedule_masking(ctc_raw_output_list, uniform=True,
                                                                       mask_token=self.text_mask_token)
            masked_ctc_raw_output = pad_sequence(masked_ctc_raw_output,
                                                 batch_first=True, padding_value=self.text_pad_id)
            ctc_raw_output_embeddings = self.text_embedding(masked_ctc_raw_output)
            ctc_correction_input_embeddings = ctc_raw_output_embeddings + speech_embeddings

        # Align TTS task inputs
        # Multiply the attention matrix with the text embeddings to expand the embeddings
        expanded_text_embeddings = torch.matmul(alignment.transpose(1, 2), text_embeddings)  # (B, T, D)
        align_input_embeddings = speech_masked_embeddings + expanded_text_embeddings  # reusing the masked speech embeddings

        # join batch-wise and run conformer and then split
        if self.use_ctc_correction:
            align_tts_and_ctc_correction_input_embeddings = torch.cat([align_input_embeddings,
                                                                       ctc_correction_input_embeddings], dim=0)
            align_tts_and_ctc_correction_attn_mask = torch.cat([speech_attn_mask, speech_attn_mask], dim=0)
        else:
            align_tts_and_ctc_correction_input_embeddings = align_input_embeddings
            align_tts_and_ctc_correction_attn_mask = speech_attn_mask

        align_tts_and_ctc_correction_out, *_ = self.conformer(align_tts_and_ctc_correction_input_embeddings,
                                                              mask=align_tts_and_ctc_correction_attn_mask)

        # Align TTS task
        align_tts_out = align_tts_and_ctc_correction_out[:align_input_embeddings.size(0)]
        align_pred_logits = self.speech_pred_head(align_tts_out[speech_masks])
        align_speech_loss = self.speech_ce_loss_fn(align_pred_logits.view(-1, align_pred_logits.size(-1)),
                                                   speech_mlm_targets.view(-1))

        if self.use_ctc_correction:
            # CTC-Correction task
            ctc_correction_out = align_tts_and_ctc_correction_out[align_input_embeddings.size(0):]
            ctc_correction_pred_logits = self.ctc_correction_pred_head(ctc_correction_out)
            ctc_correction_log_probs = F.log_softmax(ctc_correction_pred_logits, dim=-1, dtype=torch.float32)
            ctc_correction_loss = self.ctc_loss_fn(ctc_correction_log_probs.transpose(0, 1),
                                                   ctc_targets, speech_lengths, text_lengths)
        else:
            ctc_correction_loss = torch.tensor(0.0, device=self.device)

        loss = ctc_loss + length_loss + speech_mlm_loss + align_speech_loss + ctc_correction_loss

        return SpeechTextMultiTaskModelOutput(
            loss=loss,
            ctc_text_loss=ctc_loss.detach(),
            length_loss=length_loss.detach(),
            speech_mlm_loss=speech_mlm_loss.detach(),
            align_speech_loss=align_speech_loss.detach(),
            ctc_correction_loss=ctc_correction_loss.detach(),
        )

    def cosine_schedule_masking(
            self, input_sequences: List[torch.Tensor], p: torch.Tensor | None = None, uniform: bool = False,
            mask_token: int = 0
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Mask the input sequences using cosine schedule masking.
        uniformly sample from [0, pi/2] and apply cosine function to get masking probabilities
        mean of cosine function between 0 and pi/2 is 2/pi~0.64, so on average 64% of the tokens will be masked

        :param input_sequences: List[torch.Tensor], list of input sequences
        :param p: torch.Tensor, masking probabilities
        :param uniform: bool, whether to sample uniformly from (0, 1)
        :param mask_token: int, mask token
        :return: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]
        """

        if p is None:
            b = len(input_sequences)
            if uniform:
                p = torch.empty(b, device=self.device).uniform_(0, 1)
            else:
                u = torch.empty(b, device=self.device).uniform_(0, math.pi / 2)
                p = torch.cos(u)

        masks = []
        masked_sequences = []
        for i, x in enumerate(input_sequences):
            bernoulli_samples = torch.bernoulli(p[i].expand_as(x))
            mask = bernoulli_samples.bool()
            masked_sequence = torch.where(mask, mask_token, x)
            masks.append(mask)
            masked_sequences.append(masked_sequence)

        return masked_sequences, masks, p

    def _normalize(self, text):
        # return normalizer.normalize(normalize_text(text, SYMBOLS))
        return english_cleaners(text)

    def infer_special(self, text=None, semantic_tokens=None,
                      t2s_iters=10, t2s_temperature=1.0,
                      s2t_iters=10, s2t_temperature=1.0,
                      s2t_proportion=0.5,
                      s2t_ctc_only=False,
                      s2t_threshold=None,
                      t2s_normalize=True,
                      t2s_use_cfg=True,
                      t2s_use_mlm=True,
                      t2s_cfg_initial_weight=0.1, t2s_cfg_final_weight=2.0,
                      t2s_mlm_initial_weight=1.0, t2s_mlm_final_weight=0.1,
                      length_multiplier=1.0):
        """
        Inference function
        :param text:
        :param semantic_tokens:
        :param t2s_iters:
        :param t2s_temperature:
        :param s2t_iters:
        :param s2t_temperature:
        :param s2t_proportion:
        :param s2t_threshold:
        :param t2s_cfg_initial_weight:
        :param t2s_cfg_final_weight:
        :param s2t_ctc_only:
        :param t2s_normalize:
        :param t2s_use_cfg:
        :param t2s_use_mlm:
        :param t2s_mlm_initial_weight:
        :param t2s_mlm_final_weight:

        :return:
        """
        speech_pred_tokens = None
        ctc_text_pred = None
        corrected_text_pred = None
        assert text is not None or semantic_tokens is not None, "Either text or semantic tokens must be provided."
        if text is not None:
            if t2s_normalize:
                text = self._normalize(text)
            text_tokens = self.tokenizer.encode(text, num_special_token=self.num_special_tokens, device=self.device)[0]
            speech_pred_tokens = self.infer_text_to_speech(text_tokens, pred_iters=t2s_iters,
                                                           temperature=t2s_temperature,
                                                           cfg_initial_weight=t2s_cfg_initial_weight,
                                                           cfg_final_weight=t2s_cfg_final_weight,
                                                           use_mlm=t2s_use_mlm,
                                                           mlm_initial_weight=t2s_mlm_initial_weight,
                                                           mlm_final_weight=t2s_mlm_final_weight,
                                                           use_cfg=t2s_use_cfg,
                                                           length_multiplier=length_multiplier
                                                           )
            speech_pred_tokens = torch.tensor(self.speech_tokenizer.decode(speech_pred_tokens.unbind(0)),
                                              device=self.device)

        if semantic_tokens is not None:
            if semantic_tokens.dim() == 2:
                semantic_tokens = semantic_tokens.squeeze(0)
            semantic_tokens = self.speech_tokenizer.encode([semantic_tokens.tolist()], device=self.device)[0]

            if s2t_ctc_only or not self.use_ctc_correction:
                ctc_text_tokens = self.infer_ctc(semantic_tokens)
            else:
                if s2t_threshold is None:
                    corrected_text_tokens, ctc_text_tokens = self.infer_speech_to_text(semantic_tokens,
                                                                                       pred_iters=s2t_iters,
                                                                                       temperature=s2t_temperature,
                                                                                       s2t_proportion=s2t_proportion,
                                                                                       s2t_threshold=s2t_threshold)
                else:
                    corrected_text_tokens, ctc_text_tokens = self.infer_speech_to_text_confidence_masking(
                        semantic_tokens,
                        pred_iters=s2t_iters,
                        temperature=s2t_temperature,
                        s2t_threshold=s2t_threshold)

                # text_tokens = self.infer_ctc(semantic_tokens)
                corrected_text_pred = self.tokenizer.decode(corrected_text_tokens,
                                                            num_special_token=self.num_special_tokens)[0]

            if ctc_text_tokens is not None:
                ctc_text_pred = self.tokenizer.decode(ctc_text_tokens,
                                                      num_special_token=self.num_special_tokens)[0]
        return speech_pred_tokens, ctc_text_pred, corrected_text_pred

    def infer_speech_to_text(self, speech_tokens, pred_iters=10, temperature=1.0, s2t_proportion=0.5,
                             s2t_threshold=None):
        """
        Inference function
        :param speech_tokens:
        :param pred_iters:
        :param temperature:
        :param s2t_proportion:
        :param s2t_threshold:
        :return:
        """
        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        else:
            assert speech_tokens.shape[0] == 1, "Batched inference not supported yet."

        speech_embeddings = self.speech_embedding(speech_tokens)

        ctc_pred_ids = None
        if s2t_proportion != 1.0:
            encoder_out, *_ = self.conformer(speech_embeddings.clone())

            ctc_logits = self.ctc_text_pred_head(encoder_out)
            ctc_pred_ids = ctc_logits.argmax(dim=-1)

            ctc_pred_ids = ctc_pred_ids - 1
            blank_mask = ctc_pred_ids == -1
            ctc_pred_ids = ctc_pred_ids.masked_fill(blank_mask, self.text_fill_token)
            ctc_confidence = ctc_logits.softmax(dim=-1).max(dim=-1).values

            flat_confidences = ctc_confidence.flatten()

            if s2t_threshold is None:
                s2t_threshold = torch.quantile(flat_confidences, s2t_proportion)
            mask = ctc_confidence < s2t_threshold
            initial_ratio = mask.sum().float() / mask.numel()
            sampled_tokens = torch.where(mask, self.text_mask_token, ctc_pred_ids).long()
        else:
            mask = torch.ones_like(speech_tokens, dtype=torch.bool)
            sampled_tokens = torch.ones_like(speech_tokens, dtype=torch.long) * self.text_mask_token
            initial_ratio = 1.0

        gumbel_distribution = torch.distributions.gumbel.Gumbel(
            torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device)
        )

        mask_ratios = [math.cos(math.pi / 2.0 * ((t + 1) / pred_iters)) * initial_ratio for t in range(pred_iters)]
        initial_mask_num_tokens = mask.sum(dim=-1)
        # print(initial_mask_num_tokens)
        # print(initial_ratio, mask_ratios)
        if initial_ratio == 0.0:
            mask_ratios = []

        for i, mask_ratio in enumerate(mask_ratios):
            # Conditional output (with speech tokens)
            text_embeddings = self.text_embedding(sampled_tokens)
            input_embeddings = speech_embeddings + text_embeddings
            encoder_out, *_ = self.conformer(input_embeddings)
            output_logits = self.ctc_correction_pred_head(encoder_out)

            if i == pred_iters - 1:
                sampled_ids = output_logits.argmax(dim=-1)
                blank_mask = sampled_ids == self.text_ctc_blank_token
                sampled_ids = sampled_ids - 1
                sampled_ids = sampled_ids.masked_fill(blank_mask, self.text_fill_token)
                sampled_tokens = torch.where(mask, sampled_ids, sampled_tokens)
            else:
                cur_temperature = temperature * mask_ratio
                sampled_ids = output_logits.argmax(dim=-1)
                blank_mask = sampled_ids == self.text_ctc_blank_token
                sampled_ids_adjusted = sampled_ids - 1
                sampled_ids_adjusted = sampled_ids_adjusted.masked_fill(blank_mask, self.text_fill_token)
                sampled_tokens = torch.where(mask, sampled_ids_adjusted, sampled_tokens)

                mask_len = torch.floor(initial_mask_num_tokens.float() * mask_ratio).long()
                mask_len = torch.maximum(
                    torch.tensor(1, device=self.device), torch.minimum(mask_len, initial_mask_num_tokens)
                )

                probs = F.softmax(output_logits, dim=-1)
                selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)
                selected_probs = torch.where(mask, selected_probs, torch.inf)

                next_mask = random_topk_mask(
                    mask_len, selected_probs, gumbel_distribution, temperature=cur_temperature
                )

                sampled_tokens = torch.where(next_mask, self.text_mask_token, sampled_tokens)

                mask = next_mask

        # Apply CTC decoding to each sequence in the batch and return list of tensors
        pred_ids = sampled_tokens.unbind(dim=0)  # Unbind the batch dimension
        decoded_sequences = [ctc_decode(pred_id, blank_token=self.text_fill_token) for pred_id in pred_ids]

        ctc_decoded_sequences = None
        if ctc_pred_ids is not None:
            ctc_pred_ids = ctc_pred_ids.unbind(dim=0)  # Unbind the batch dimension
            ctc_decoded_sequences = [ctc_decode(pred_id, blank_token=self.text_fill_token) for pred_id in ctc_pred_ids]

        return decoded_sequences, ctc_decoded_sequences

    def infer_speech_to_text_confidence_masking(self, speech_tokens, pred_iters=10, temperature=1.0,
                                                s2t_threshold=0.8):
        """
        Inference function
        :param speech_tokens:
        :param pred_iters:
        :param temperature:
        :param s2t_threshold:
        :return:
        """
        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        else:
            assert speech_tokens.shape[0] == 1, "Batched inference not supported yet."

        speech_embeddings = self.speech_embedding(speech_tokens)

        encoder_out, *_ = self.conformer(speech_embeddings.clone())

        ctc_logits = self.ctc_text_pred_head(encoder_out)
        ctc_pred_ids = ctc_logits.argmax(dim=-1)

        ctc_pred_ids = ctc_pred_ids - 1
        blank_mask = ctc_pred_ids == -1
        ctc_pred_ids = ctc_pred_ids.masked_fill(blank_mask, self.text_fill_token)
        ctc_confidence = ctc_logits.softmax(dim=-1).max(dim=-1).values

        mask = ctc_confidence < s2t_threshold
        # initial_ratio = mask.sum().float() / mask.numel()
        sampled_tokens = torch.where(mask, self.text_mask_token, ctc_pred_ids).long()

        # gumbel_distribution = torch.distributions.gumbel.Gumbel(
        #     torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device)
        # )

        temperatures = torch.linspace(temperature, 0.1, pred_iters)

        for i in range(pred_iters):
            # Conditional output (with speech tokens)
            text_embeddings = self.text_embedding(sampled_tokens)
            input_embeddings = speech_embeddings + text_embeddings
            encoder_out, *_ = self.conformer(input_embeddings)
            output_logits = self.ctc_correction_pred_head(encoder_out)

            if i == pred_iters - 1:
                sampled_ids = output_logits.argmax(dim=-1)
                blank_mask = sampled_ids == self.text_ctc_blank_token
                sampled_ids = sampled_ids - 1
                sampled_ids = sampled_ids.masked_fill(blank_mask, self.text_fill_token)
                sampled_tokens = torch.where(mask, sampled_ids, sampled_tokens)
            else:
                cur_temperature = temperatures[i]
                scaled_output_logits = output_logits / cur_temperature
                sampled_ids = scaled_output_logits.argmax(dim=-1)
                blank_mask = sampled_ids == self.text_ctc_blank_token
                sampled_ids_adjusted = sampled_ids - 1
                sampled_ids_adjusted = sampled_ids_adjusted.masked_fill(blank_mask, self.text_fill_token)
                sampled_tokens = torch.where(mask, sampled_ids_adjusted, sampled_tokens)
                scaled_confidences = scaled_output_logits.softmax(dim=-1).max(dim=-1).values
                # next_mask should only update tokens that were originally masked
                next_mask = torch.where(mask, scaled_confidences < s2t_threshold, mask)
                # next_mask = scaled_confidences < s2t_threshold

                if next_mask.sum() == 0:
                    break

                sampled_tokens = torch.where(next_mask, self.text_mask_token, sampled_tokens)

                mask = next_mask

        # Apply CTC decoding to each sequence in the batch and return list of tensors
        pred_ids = sampled_tokens.unbind(dim=0)  # Unbind the batch dimension
        decoded_sequences = [ctc_decode(pred_id, blank_token=self.text_fill_token) for pred_id in pred_ids]

        ctc_decoded_sequences = None
        if ctc_pred_ids is not None:
            ctc_pred_ids = ctc_pred_ids.unbind(dim=0)  # Unbind the batch dimension
            ctc_decoded_sequences = [ctc_decode(pred_id, blank_token=self.text_fill_token) for pred_id in ctc_pred_ids]

        return decoded_sequences, ctc_decoded_sequences

    def infer_ctc(self, speech_tokens):
        """
        Inference function
        :param speech_tokens:
        :return:
        """
        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        else:
            assert speech_tokens.shape[0] == 1, "Batched inference not supported yet."

        speech_embeddings = self.speech_embedding(speech_tokens)
        encoder_out, *_ = self.conformer(speech_embeddings)
        pred_logits = self.ctc_text_pred_head(encoder_out)

        pred_ids = pred_logits.argmax(dim=-1)
        blank_mask = pred_ids == self.text_ctc_blank_token
        pred_ids = pred_ids - 1
        pred_ids = pred_ids.masked_fill(blank_mask, self.text_fill_token)

        ctc_decoded_sequences = [ctc_decode(pred_id, blank_token=self.text_fill_token) for pred_id in pred_ids]

        return ctc_decoded_sequences

    def infer_text_to_speech(self, text_tokens, pred_iters=10, temperature=1.0,
                             cfg_initial_weight=3.0, cfg_final_weight=0.1,
                             use_mlm=False, mlm_initial_weight=1.0, mlm_final_weight=0.1,
                             length_multiplier=1.0,
                             use_cfg=False):
        """
        Inference function
        :param text_tokens:
        :param pred_iters:
        :param temperature:
        :param cfg_initial_weight:
        :param cfg_final_weight:
        :param use_mlm:
        :param use_cfg:
        :param mlm_initial_weight:
        :param mlm_final_weight:
        :return:
        """
        if len(text_tokens.shape) == 1:
            text_tokens = text_tokens.unsqueeze(0)
        else:
            assert text_tokens.shape[0] == 1, "Batched inference not supported yet."

        length_predictor_input_embeddings = self.text_embedding(text_tokens)
        length_out, *_ = self.conformer(length_predictor_input_embeddings)
        length_preds = self.length_prediction_head(length_out).squeeze(-1)
        lengths = length_preds.exp().ceil().long().squeeze(0)
        lengths = (lengths * length_multiplier).ceil().long()

        # repeat each token in length_predictor_inputs by the corresponding length
        tts_inputs = torch.repeat_interleave(text_tokens, lengths, dim=1)

        text_embeddings = self.text_embedding(tts_inputs)

        length = lengths.sum()

        sampled_tokens = torch.ones(1, length, device=self.device, dtype=torch.long) * self.speech_mask_token
        mask = torch.ones_like(sampled_tokens, dtype=torch.bool)

        gumbel_distribution = torch.distributions.gumbel.Gumbel(
            torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device)
        )

        mask_ratios = [math.cos(math.pi / 2.0 * ((t + 1) / pred_iters)) for t in range(pred_iters)]
        initial_mask_num_tokens = mask.sum(dim=-1)

        cfg_weights = None
        mlm_weights = None
        output_logits_uncond = None
        if use_cfg and self.use_speech_mlm:
            # Interpolation between the initial and final CFG weights
            cfg_weights = torch.linspace(cfg_initial_weight, cfg_final_weight, pred_iters)
            all_mask_embeddings = self.speech_embedding(sampled_tokens)
            encoder_out_uncond, *_ = self.conformer(all_mask_embeddings)
            output_logits_uncond = self.speech_pred_head(encoder_out_uncond)

        if use_mlm and self.use_speech_mlm:
            # Interpolation between the initial and final MLM weights
            mlm_weights = torch.linspace(mlm_initial_weight, mlm_final_weight, pred_iters)

        for i, mask_ratio in enumerate(mask_ratios):
            # Conditional output (with text tokens)
            speech_embeddings = self.speech_embedding(sampled_tokens)
            input_embeddings_cond = text_embeddings.clone() + speech_embeddings.clone()
            encoder_out_cond, *_ = self.conformer(input_embeddings_cond)
            output_logits_cond = self.speech_pred_head(encoder_out_cond)

            output_logits = output_logits_cond
            # mlm blending
            if use_mlm and self.use_speech_mlm:
                input_embeddings_mlm = speech_embeddings.clone()  # Unconditional (without text embeddings)
                encoder_out_mlm, *_ = self.conformer(input_embeddings_mlm)
                output_logits_mlm = self.speech_pred_head(encoder_out_mlm)
                # output_logits = mlm_weights[i] * output_logits_mlm + (1 - mlm_weights[i]) * output_logits_cond
                # output_logits = output_logits_mlm + (output_logits_cond - output_logits_mlm) * mlm_initial_weight
                output_logits = (1 + mlm_initial_weight) * output_logits_cond - mlm_initial_weight * output_logits_mlm

            if use_cfg and self.use_speech_mlm:
                # Classifier-Free Guidance blending
                output_logits = (1 + cfg_initial_weight) * output_logits_cond - cfg_initial_weight * output_logits_uncond
                # output_logits = (1 + cfg_weights[i]) * output_logits_cond - cfg_weights[i] * output_logits_uncond
                # output_logits = output_logits_uncond + (output_logits_cond - output_logits_uncond) * cfg_initial_weight

            if i == pred_iters - 1:
                sampled_ids = output_logits.argmax(dim=-1)
                sampled_tokens = torch.where(mask, sampled_ids, sampled_tokens)
            else:
                cur_temperature = temperature * mask_ratio
                sampled_ids = torch.distributions.categorical.Categorical(
                    logits=output_logits / cur_temperature).sample()
                # sampled_ids = torch.distributions.categorical.Categorical(
                #     logits=output_logits).sample()
                sampled_tokens = torch.where(mask,
                                             sampled_ids,
                                             sampled_tokens)

                mask_len = torch.floor(initial_mask_num_tokens.float() * mask_ratio).long()
                mask_len = torch.maximum(
                    torch.tensor(1, device=self.device), torch.minimum(mask_len, initial_mask_num_tokens)
                )

                probs = F.softmax(output_logits, dim=-1)
                selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)
                selected_probs = torch.where(mask, selected_probs, torch.inf)

                next_mask = random_topk_mask(
                    mask_len, selected_probs, gumbel_distribution, temperature=cur_temperature
                )

                sampled_tokens = torch.where(next_mask, self.speech_mask_token, sampled_tokens)

                mask = next_mask

        return sampled_tokens
