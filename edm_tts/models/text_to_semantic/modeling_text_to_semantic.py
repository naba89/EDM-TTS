import math
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.utils import ModelOutput

from edm_tts.models.conformer.conformer import Conformer
from edm_tts.models.text_to_semantic.configuration import TextToSemanticWLenConfig
from edm_tts.utils.utils import random_topk_mask


@dataclass
class TextToSemanticWLenOutput(ModelOutput):
    loss: torch.Tensor = None
    ce_loss: torch.Tensor = None
    length_loss: torch.Tensor = None
    prompt_kl_loss: torch.Tensor = None
    speech_pred_tokens: torch.Tensor = None


class TextToSemanticWLen(PreTrainedModel, ModuleUtilsMixin):
    config_class = TextToSemanticWLenConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.register_buffer("text_token", torch.tensor([config.special_tokens["text"]], dtype=torch.long))
        self.register_buffer("speech_token", torch.tensor([config.special_tokens["speech"]], dtype=torch.long))
        self.register_buffer("sep_token", torch.tensor([config.special_tokens["sep"]], dtype=torch.long))
        self.register_buffer("pad_token", torch.tensor([config.special_tokens["pad"]], dtype=torch.long))
        self.register_buffer("mask_token", torch.tensor([config.special_tokens["mask"]], dtype=torch.long))
        self.register_buffer("false", torch.tensor([0], dtype=torch.bool))

        self.num_special_tokens = len(config.special_tokens)

        self.total_num_tokens = config.text_vocab_size + config.semantic_vocab_size + self.num_special_tokens
        self.pad_token_id = config.special_tokens["pad"]

        self.dim = config.hidden_size

        self.input_embedding = nn.Embedding(self.total_num_tokens, self.dim, padding_idx=self.pad_token_id)

        self.conformer = Conformer(dim=self.dim, **config.main_encoder_args)

        self.length_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.length_predictor = Conformer(dim=self.dim, **config.length_predictor_args)

        self.pred_transform = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.GELU(approximate="tanh"), nn.LayerNorm(self.dim)
        )
        self.pred_head = nn.Linear(self.dim, config.semantic_vocab_size, bias=True)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        self.length_pred_head = nn.Linear(self.dim, 1, bias=True)
        self.length_loss_fn = nn.L1Loss()

    def cosine_schedule_mask(self, feature_length):
        """
        Args:
            feature_length: length of the feature sequence
        """
        u = torch.empty(1, device=self.device).uniform_(0, math.pi / 2)
        p = torch.cos(u)
        p = p.expand(feature_length)
        bernoulli_samples = torch.bernoulli(p)
        mask = bernoulli_samples.bool()

        return mask

    def random_masking_and_input_prep(self, x):

        # Split the input into text and speech token sequences

        text_sequences = []
        speech_length_targets = []

        combined_input = x.clone()
        combined_mask = x.new_zeros(x.size(), dtype=torch.bool)

        for i, sequence in enumerate(x):
            text_start = (sequence == self.text_token).nonzero(as_tuple=True)[0][0]
            speech_start = (sequence == self.speech_token).nonzero(as_tuple=True)[0][0]
            sep_indices = (sequence == self.sep_token).nonzero(as_tuple=True)[0]
            if len(sep_indices) != 2:
                raise ValueError("Each sequence must contain exactly two [SEP] tokens.")
            text_seq = sequence[text_start + 1: sep_indices[0]]
            speech_seq = sequence[speech_start + 1: sep_indices[1]]

            mask = self.cosine_schedule_mask(len(speech_seq))
            masked_speech_seq = torch.where(mask, self.mask_token.expand_as(speech_seq), speech_seq)
            combined_input[i, speech_start + 1: sep_indices[1]] = masked_speech_seq
            combined_mask[i, speech_start + 1: sep_indices[1]] = mask

            text_sequences.append(text_seq)
            speech_length_targets.append(len(speech_seq))

        text_sequences = pad_sequence(text_sequences, batch_first=True, padding_value=self.pad_token_id)
        text_attention_mask = text_sequences != self.pad_token_id

        speech_length_targets = torch.tensor(speech_length_targets, device=self.device).float().log()

        return text_sequences, text_attention_mask, combined_input, combined_mask, speech_length_targets

    def forward(self, input_ids, attention_mask, **kwargs):
        b = input_ids.size(0)

        (text_sequences, text_attention_mask,
         masked_input, mask, length_targets) = self.random_masking_and_input_prep(input_ids)

        text_embeddings = self.input_embedding(text_sequences)
        length_predictor_input = torch.cat([self.length_token.expand(b, -1, -1), text_embeddings], dim=1)
        length_predictor_attention_mask = torch.cat([torch.ones(b, 1, device=self.device, dtype=torch.bool),
                                                     text_attention_mask], dim=1)  # for the length token
        length_out, *_ = self.length_predictor(length_predictor_input,
                                               mask=length_predictor_attention_mask, return_attn=False)
        length_out = length_out[:, 0]
        length_pred = self.length_pred_head(length_out).squeeze(-1)

        main_encoder_inputs = self.input_embedding(masked_input)

        pred_logits = self.embeddings_to_logits(main_encoder_inputs, attention_mask=attention_mask, mask=mask)

        # shift the targets to original semantic vocab size and get only masked tokens
        targets = input_ids[mask]
        targets = targets - self.num_special_tokens - self.config.text_vocab_size

        ce_loss = self.loss_fn(pred_logits.view(-1, pred_logits.size(-1)), targets.view(-1))
        length_loss = self.length_loss_fn(length_pred, length_targets)

        loss = ce_loss + length_loss

        return TextToSemanticWLenOutput(
            loss=loss,
            ce_loss=ce_loss.detach(),
            length_loss=length_loss.detach(),
        )

    def embeddings_to_logits(
            self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None,
            mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Convert embeddings to logits
        :param embeddings: torch.Tensor, shape: [batch_size, seq_len, hidden_size]
        :param attention_mask: torch.Tensor, shape: [batch_size, seq_len]
        :param mask: torch.Tensor, shape: [batch_size, seq_len]
        :return: torch.Tensor, shape: [-1, total_num_tokens]
        """
        out, *_ = self.conformer(embeddings, mask=attention_mask, return_attn=False)
        if mask is not None:
            out = out[mask]
        pred_transformed = self.pred_transform(out)
        pred_logits = self.pred_head(pred_transformed)
        return pred_logits

    def extract_features(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
            output_layer_idx: int | None = None,
            return_attn: bool = False
    ):
        """
        Extract features from the model
        :param input_ids: torch.Tensor, shape: [batch_size, seq_len]
        :param attention_mask: torch.Tensor, shape: [batch_size, seq_len]
        :param output_layer_idx: int, index of the output layer
        :param return_attn: bool, whether to return attention weights
        :return: torch.Tensor, shape: [batch_size, seq_len, hidden_size]
        """

        x = self.input_embedding(input_ids)

        features, attn = self.conformer(x, mask=attention_mask, output_layer_idx=output_layer_idx,
                                        return_attn=return_attn)

        return features, attn

    def infer(self, text, pred_iters=10, temperature=1.0, gt_length=None, **kwargs):
        """
        Inference function
        :param text: str, text input
        :param pred_iters: int, number of prediction iterations
        :param temperature: float, temperature for sampling
        :param gt_length: int, ground truth length
        :return: TextToSemanticWLenOutput
        """
        text_tokens = torch.tensor(list(text.encode("utf-8")), dtype=torch.long,
                                   device=self.device) + self.num_special_tokens

        if gt_length is not None:
            length_pred = gt_length
        else:
            text_embeddings = self.input_embedding(text_tokens).unsqueeze(0)
            length_predictor_input = torch.cat([self.length_token, text_embeddings], dim=1)
            length_out, *_ = self.length_predictor(length_predictor_input, return_attn=False)
            length_out = length_out[:, 0]
            length_pred = self.length_pred_head(length_out).squeeze(-1).exp().ceil().long()

        input_ids = torch.cat([
            self.text_token, text_tokens, self.sep_token,
            self.speech_token, self.mask_token.repeat(length_pred), self.sep_token
        ]).unsqueeze(0)

        full_mask = torch.cat([
            self.false,  # [text] token
            self.false.repeat(len(text_tokens)),  # text tokens
            self.false,  # [sep] token
            self.false,  # [speech] token
            torch.ones(length_pred, device=self.device).bool(),  # masked speech tokens
            self.false  # [sep] token
        ]).bool().unsqueeze(0)

        sampled_tokens = input_ids.clone()

        # create gumbel distribution on the device of the acoustic tokens
        gumbel_distribution = torch.distributions.gumbel.Gumbel(
            torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device)
        )

        mask_ratios = [math.cos(math.pi / 2.0 * ((t + 1) / pred_iters)) for t in range(pred_iters)]

        mask = full_mask.clone()
        initial_mask_num_tokens = mask.sum(dim=-1)

        for i, mask_ratio in enumerate(mask_ratios):

            input_embs = self.input_embedding(sampled_tokens)
            output_logits = self.embeddings_to_logits(input_embs)

            if i == pred_iters - 1:
                sampled_tokens = output_logits.argmax(dim=-1)
                sampled_tokens = torch.where(full_mask, sampled_tokens, input_ids)
            else:
                sampled_tokens = torch.distributions.categorical.Categorical(logits=output_logits).sample()

                mask_len = torch.floor(initial_mask_num_tokens.float() * mask_ratio).long()

                # max(1, min(how many unknown tokens, how many tokens we want to sample))
                mask_len = torch.maximum(
                    torch.tensor(1, device=self.device), torch.minimum(mask_len, initial_mask_num_tokens)
                )

                probs = F.softmax(output_logits, dim=-1)

                # get probability for selected tokens in categorical call, also for already sampled ones
                selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_tokens, -1), -1), -1)

                # ignore tokens which are already sampled
                selected_probs = torch.where(mask, selected_probs, torch.inf)

                next_mask = random_topk_mask(
                    mask_len, selected_probs, gumbel_distribution, temperature=temperature * mask_ratio
                )

                sampled_tokens = torch.where(next_mask, self.mask_token,
                                             sampled_tokens + self.num_special_tokens + self.config.text_vocab_size)
                sampled_tokens = torch.where(full_mask, sampled_tokens, input_ids)

                mask = next_mask

        return TextToSemanticWLenOutput(speech_pred_tokens=sampled_tokens[full_mask])
