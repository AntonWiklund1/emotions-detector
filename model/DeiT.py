import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import constants

torch.manual_seed(42)

image_size = constants.image_size
patch_size = constants.patch_size

class DistillationToken(nn.Module):
    def __init__(self, embed_dim):
        super(DistillationToken, self).__init__()
        self.dist_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        # x is the last_hidden_state from ViTModel with shape [batch_size, num_patches, embed_dim]
        return torch.cat([x, self.dist_token.expand(x.size(0), -1, -1)], dim=1)

class DeiT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim):
        super(DeiT, self).__init__()
        self.config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_hidden_layers=4,  # Number of transformer layers
            hidden_size=embed_dim,
            num_attention_heads=4,
            intermediate_size=embed_dim * 4,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.4,
            attention_probs_dropout_prob=0.4,
            num_channels=1
        )

        self.vit = ViTModel(self.config)
        self.dist_token = DistillationToken(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dist_classifier = nn.Linear(embed_dim, num_classes)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        embeddings = outputs.last_hidden_state

        # Apply layer normalization before adding the distillation token
        embeddings = self.layer_norm(embeddings)

        embeddings = self.dist_token(embeddings)
        
        # Apply layer normalization after adding the distillation token
        embeddings = self.layer_norm(embeddings)

        class_logits = self.classifier(embeddings[:, 0])
        dist_logits = self.dist_classifier(embeddings[:, -1])  # Use the last token for distillation

        return class_logits, dist_logits