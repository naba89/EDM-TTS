import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.utils import ModelOutput

from edm_tts.models.dac import DAC
from edm_tts.models.injection_conformer.injection_conformer_wrapper  import InjectionConformerWrapper
from edm_tts.models.injection_conformer.configuration import InjectionConformerConfig
from edm_tts.utils.utils import random_topk_mask


@dataclass
class InjectionConformerOutput(ModelOutput):
    loss: torch.Tensor = None
    output_acoustic_codes: torch.Tensor = None
    target_acoustic_codes: torch.Tensor = None


class InjectionConformerModel(PreTrainedModel, ModuleUtilsMixin):
    config_class = InjectionConformerConfig

    def __init__(self, config):
        super().__init__(config)

        # self.acoustic_model = DAC.from_pretrained(config.acoustic_model_path).eval()
        self.acoustic_model = DAC.from_pretrained("subatomicseer/acoustic_tokenizer").eval()

        self.acoustic_size = self.acoustic_model.latent_dim
        self.num_codevectors = self.acoustic_model.codebook_size
        self.num_quantizers = self.acoustic_model.n_codebooks
        for p in self.acoustic_model.parameters():
            p.requires_grad = False

        self.acoustic_feat_proj = nn.Sequential(
            nn.Linear(self.acoustic_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

        self.semantic_embedding = nn.Embedding(config.num_semantic_tokens, config.hidden_size)
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        self.injection_layers = config.injection_layers

        self.encoder = InjectionConformerWrapper(
            dim=config.hidden_size,
            injection_layers=config.injection_layers,
            injection_dim=self.acoustic_size,
            num_quantizers=self.num_quantizers,
            num_codevectors=self.num_codevectors,
            residual=config.residual,
            use_injection=config.use_injection,
            **config.encoder_config
        )

        self.loss_all = config.loss_all

    def cosine_schedule_mask(self, feature_length, batch_size):
        """
        Args:
            feature_length: length of the feature sequence
            batch_size: batch size
        """
        u = torch.empty(batch_size, device=self.device).uniform_(0, math.pi / 2)
        p = torch.cos(u)
        p = p.unsqueeze(1).expand(batch_size, feature_length)
        bernoulli_samples = torch.bernoulli(p)
        mask = bernoulli_samples.bool()

        return mask

    def forward(self, acoustic_tokens, semantic_tokens):
        """
        Args:
            acoustic_tokens: (b, q, t) tensor of acoustic tokens
            semantic_tokens: (b, t) tensor of semantic tokens
        """

        assert acoustic_tokens.shape[-1] == semantic_tokens.shape[
            -1], "Acoustic and semantic tokens must have same length"

        acoustic_target_tokens = acoustic_tokens.clone()

        semantic_features = self.semantic_embedding(semantic_tokens)

        acoustic_features_unreduced = self.acoustic_model.codes_to_features_unreduced(acoustic_tokens)  # bqdt

        acoustic_features = self.acoustic_feat_proj(acoustic_features_unreduced[:, 0].transpose(1, 2))

        batch_size, feature_length, dim = semantic_features.shape

        mask_time_indices = self.cosine_schedule_mask(feature_length, batch_size)

        encoder_input = torch.where(mask_time_indices[:, :, None],
                                    semantic_features + self.mask_token.expand(batch_size, feature_length, -1),
                                    semantic_features + acoustic_features)

        # Inject coarse acoustic features
        injections = [acoustic_features_unreduced[:, :i + 1].sum(1).transpose(1, 2)
                      for i in range(len(self.injection_layers))]

        all_logits = self.encoder(
            x=encoder_input,
            injections=injections,
            acoustic_model=self.acoustic_model,
        )  # (b, q, n, l)

        if not self.loss_all:
            all_logits = all_logits.masked_select(mask_time_indices[:, None, :, None]).view(-1, self.num_codevectors)
            acoustic_target_for_loss = torch.masked_select(acoustic_target_tokens, mask_time_indices[:, None, :]).view(-1)
        else:
            acoustic_target_for_loss = acoustic_target_tokens.view(-1)

        logits_for_loss = rearrange(all_logits, '... l -> (...) l')

        loss = F.cross_entropy(logits_for_loss, acoustic_target_for_loss, reduction='mean')

        output_codes = all_logits.argmax(dim=-1)

        return InjectionConformerOutput(
            loss=loss,
            output_acoustic_codes=output_codes,
            target_acoustic_codes=acoustic_target_tokens
        )

    def infer_special(self, semantic_tokens, acoustic_prompt_tokens, semantic_prompt_tokens, steps=1, temperature=1.0):
        """
        Args:
            semantic_tokens: (b, t) tensor of semantic tokens
            acoustic_prompt_tokens: (b, q, t) tensor of acoustic prompt tokens
            semantic_prompt_tokens: (b, q, t) tensor of semantic prompt tokens
            steps: number of prediction steps default 1
            temperature: temperature for sampling logits default 1.0
        """
        semantic_features = self.semantic_embedding(semantic_tokens)

        prompt_injections = mask_time_indices = first_level_ids = None
        prompt_length = 0
        encoder_input = semantic_features + self.mask_token.expand(semantic_features.shape[0],
                                                                   semantic_features.shape[1], -1)

        if acoustic_prompt_tokens is not None and semantic_prompt_tokens is not None:
            semantic_prompt_features = self.semantic_embedding(semantic_prompt_tokens)

            acoustic_prompt_features_unreduced = self.acoustic_model.codes_to_features_unreduced(acoustic_prompt_tokens)
            acoustic_prompt = self.acoustic_feat_proj(acoustic_prompt_features_unreduced[:, 0].transpose(1, 2))

            num_injections = min(len(self.injection_layers), acoustic_prompt_tokens.shape[1])
            injections = [acoustic_prompt_features_unreduced[:, :i + 1].sum(1).transpose(1, 2)
                          for i in range(num_injections)]

            prompt_length = acoustic_prompt.shape[1]

            zeros_padding = torch.zeros(encoder_input.shape[0], semantic_features.shape[1],
                                        injections[0].shape[-1],
                                        device=encoder_input.device, dtype=encoder_input.dtype)

            prompt_injections = [torch.cat([injection, zeros_padding], dim=1) for injection in injections]

            encoder_input = torch.cat([semantic_prompt_features + acoustic_prompt, encoder_input], dim=1)

            mask_time_indices = torch.zeros(encoder_input.shape[0], encoder_input.shape[1], dtype=torch.bool,
                                            device=encoder_input.device)
            mask_time_indices[:, prompt_length:] = True

        if steps > 1:
            gumbel_distribution = torch.distributions.gumbel.Gumbel(
                torch.tensor([0.], device=self.device),
                torch.tensor([1.], device=self.device))

            mask_ratios = [math.cos(math.pi / 2. * ((t + 1) / steps)) for t in range(steps)]

            mask = torch.ones(encoder_input.shape[0], encoder_input.shape[1] - prompt_length, device=self.device,
                              dtype=torch.bool)
            initial_mask_num_tokens = mask.sum(dim=-1)

            for i, mask_ratio in enumerate(mask_ratios):
                logits = self.encoder.forward_first_level(encoder_input.clone(), mask_time_indices=mask_time_indices)

                if i == steps - 1:
                    sampled_ids = logits.argmax(dim=-1)
                    sampled_acoustic_features = self.acoustic_model.codes_to_features(sampled_ids).transpose(1, 2)
                    sampled_acoustic_features_proj = self.acoustic_feat_proj(sampled_acoustic_features)
                    encoder_input[:, prompt_length:] = torch.where(mask[..., None],
                                                                   semantic_features + sampled_acoustic_features_proj,
                                                                   encoder_input[:, prompt_length:])
                else:
                    sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()
                    sampled_acoustic_features = self.acoustic_model.codes_to_features(sampled_ids).transpose(1, 2)
                    sampled_acoustic_features = self.acoustic_feat_proj(sampled_acoustic_features)
                    encoder_input[:, prompt_length:] = torch.where(mask[..., None],
                                                                   semantic_features + sampled_acoustic_features,
                                                                   encoder_input[:, prompt_length:])

                    mask_len = torch.floor(initial_mask_num_tokens * mask_ratio)
                    mask_len = torch.maximum(torch.ones_like(mask_len),
                                             torch.minimum(torch.sum(mask, dim=-1) - 1,
                                                           mask_len))
                    probs = F.softmax(logits[:, 0], dim=-1)

                    selected_probs = torch.squeeze(torch.take_along_dim(probs,
                                                                        torch.unsqueeze(sampled_ids[:, 0], -1),
                                                                        -1), -1)

                    selected_probs = torch.where(mask, selected_probs, torch.inf)

                    next_mask = random_topk_mask(mask_len, selected_probs, gumbel_distribution,
                                                 temperature=temperature * mask_ratio)

                    encoder_input[:, prompt_length:] = torch.where(next_mask[..., None],
                                                                   semantic_features + self.mask_token.expand(
                                                                       semantic_features.shape[0],
                                                                       semantic_features.shape[1], -1),
                                                                   encoder_input[:, prompt_length:])
                    mask = next_mask

        all_logits = self.encoder(
            x=encoder_input,
            injections=prompt_injections,
            acoustic_model=self.acoustic_model,
            mask_time_indices=mask_time_indices
        )

        output_codes = all_logits.argmax(dim=-1)

        return output_codes
