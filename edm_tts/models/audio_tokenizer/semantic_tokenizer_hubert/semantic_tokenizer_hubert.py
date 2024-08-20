import logging
import os

import torch
from einops import rearrange, repeat
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel

logging.root.setLevel(logging.ERROR)


class SemanticModelHuBERT(nn.Module):

    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ll60k",
        cluster_centers_path: str | None = None,
        output_layer: int = 18,
    ) -> None:
        super().__init__()
        self.output_layer = output_layer

        self.model = AutoModel.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        self.sample_rate = (
            self.feature_extractor.sampling_rate if hasattr(self.feature_extractor, "sampling_rate") else 16000
        )

        self.model.eval()

        if cluster_centers_path is None:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            cluster_centers_path = os.path.join(
                this_dir,
                "semantic_cluster_centers/" "hubert_large_LL60k_L18_K1024_kmeans.pt",
            )

        cluster_centers = torch.load(cluster_centers_path)
        self.register_buffer("cluster_centers", cluster_centers)

        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def codebook_size(self) -> int:
        return self.cluster_centers.shape[0]

    @property
    def downsample_factor(self) -> int:
        # todo: double check
        return 320

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def prepare_input(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = self.feature_extractor(audio.squeeze(), return_tensors="pt", sampling_rate=self.sample_rate)
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        return inputs

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """encode audio

        Args:
            audio (torch.Tensor): shape: (1, sampling_rate * second)

        Returns:
            torch.Tensor: encoded audio
        """
        inputs = self.prepare_input(audio)
        embed = self.model(**inputs, output_hidden_states=True).hidden_states[self.output_layer]
        batched_cluster_centers = repeat(self.cluster_centers, "c d -> b c d", b=embed.shape[0])
        dists = -torch.cdist(embed, batched_cluster_centers, p=2)
        clusters = dists.argmax(dim=-1)
        semantic_tokens = rearrange(clusters, "b ... -> b (...)")
        return semantic_tokens

    @torch.inference_mode()
    def encode_batch(self, input_values, attention_mask):
        embed = self.model(input_values=input_values, attention_mask=attention_mask,
                           output_hidden_states=True).hidden_states[self.output_layer]
        batched_cluster_centers = repeat(self.cluster_centers, "c d -> b c d", b=embed.shape[0])
        dists = -torch.cdist(embed, batched_cluster_centers, p=2)
        clusters = dists.argmax(dim=-1)
        semantic_tokens = rearrange(clusters, "b ... -> b (...)")
        return semantic_tokens
