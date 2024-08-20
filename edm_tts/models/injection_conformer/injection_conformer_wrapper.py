import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange, EinMix

from edm_tts.models.conformer.conformer import Conformer, exists


class InjectionConformerWrapper(Conformer):
    def __init__(self,
                 dim,
                 injection_layers,
                 injection_dim,
                 num_codevectors,
                 num_quantizers,
                 residual,
                 use_injection,  # control whether to use multi level output but without injection
                 *args, **kwargs):
        super().__init__(dim, *args, **kwargs)

        self.injection_layers = injection_layers
        self.injection_dim = injection_dim
        self.residual = residual
        self.use_injection = use_injection

        if self.use_injection:
            self.project_injection = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(injection_dim, dim),
                    nn.LayerNorm(dim),
                ) for _ in injection_layers
            ])

        self.num_codevectors = num_codevectors
        self.num_quantizers = num_quantizers
        self.remaining_quantizers = self.num_quantizers - len(self.injection_layers)

        self.fine_head = nn.Sequential(
            nn.Linear(dim, dim * self.remaining_quantizers),
            Rearrange('b n (q d) -> b n q d', q=self.remaining_quantizers),
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            EinMix(
                'b n q d -> b n q l',
                weight_shape='q d l',
                bias_shape='q l',
                q=self.num_quantizers,
                l=self.num_codevectors,
                d=dim
            ),
            Rearrange('b n q l -> b q n l')
        )

    def apply_single_to_logits(self, inp, idx):
        inp_normed = self.to_logits[0](inp)
        inp_rearranged = rearrange(inp_normed, 'b (n q) d -> b n q d', q=1)
        weight = self.to_logits[1].weight[idx]
        bias = self.to_logits[1].bias[..., idx, :]
        output = (inp_rearranged @ weight) + bias
        output = rearrange(output, 'b n q l -> b q n l')
        return output

    def forward_first_level(self, x, mask=None, mask_time_indices=None):
        seq_len = x.shape[-2]

        rotary_emb_x = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None

        first_level_logits = None
        for i, block in enumerate(self.layers):
            current_output, *_ = block(
                x,
                mask=mask,
                rotary_emb_x=rotary_emb_x,
                return_attn=False
            )
            if i in self.injection_layers:
                injection_idx = self.injection_layers.index(i)
                first_level_logits = self.apply_single_to_logits(current_output, injection_idx)
                # since we are predicting only the first level, we can break after the first injection layer
                break
            x = current_output

        if mask_time_indices is not None:
            b, _, t, d = first_level_logits.shape
            first_level_logits = torch.masked_select(first_level_logits,
                                                     mask_time_indices[:, None, :, None]).view(b, 1, -1, d)

        return first_level_logits

    def forward(self, x, mask=None, injections=None, acoustic_model=None, mask_time_indices=None):
        seq_len = x.shape[-2]

        rotary_emb_x = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None

        coarse_layer_outputs = []
        coarse_logits = []

        for i, block in enumerate(self.layers):
            current_output, *_ = block(
                x.clone(),
                mask=mask,
                rotary_emb_x=rotary_emb_x,
                return_attn=False
            )
            if i in self.injection_layers:
                injection_idx = self.injection_layers.index(i)
                residual = 0
                if len(coarse_layer_outputs) > 0 and self.residual:
                    residual = coarse_layer_outputs[-1].squeeze(2)
                coarse_layer_outputs.append(current_output.clone()[:, :, None, :])
                if self.use_injection:
                    if self.training and exists(injections):
                        current_output = (current_output
                                          + self.project_injection[injection_idx](injections[injection_idx])
                                          + residual)
                    else:
                        coarse_logits.append(self.apply_single_to_logits(current_output, injection_idx))
                        logits_until_now = torch.cat(coarse_logits, dim=1)
                        tokens_until_now = logits_until_now.argmax(dim=-1)
                        injection = acoustic_model.codes_to_features(tokens_until_now)
                        injection = rearrange(injection, 'b d n -> b n d')
                        if exists(injections) and exists(mask_time_indices):
                            injection = torch.where(mask_time_indices[:, :, None], injection, injections[injection_idx])
                        elif exists(injections) and injection_idx < len(injections):
                            injection = injections[injection_idx]
                        injection = self.project_injection[injection_idx](injection)
                        current_output = current_output + injection + residual
                else:
                    current_output = current_output + residual

            x = current_output

        if mask_time_indices is not None:
            b, n, d = x.shape
            x = torch.masked_select(x, mask_time_indices[:, :, None]).view(b, -1, d)
            coarse_layer_outputs = [torch.masked_select(coarse_layer_output,
                                                        mask_time_indices[:, :, None, None])
                                    .view(b, -1, 1, d)
                                    for coarse_layer_output in coarse_layer_outputs]

        fine_layer_output = self.fine_head(x)

        all_layer_outputs = coarse_layer_outputs + [fine_layer_output]
        all_layer_outputs = torch.cat(all_layer_outputs, dim=-2)

        logits = self.to_logits(all_layer_outputs)

        return logits
