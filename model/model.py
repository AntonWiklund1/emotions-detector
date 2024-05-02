import torch
import torch.nn as nn
from torchvision.models import resnet50
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
        return torch.cat([x, self.dist_token.repeat(x.size(0), 1, 1)], dim=1)

class DeiT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim):
        super().__init__()
        self.config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_hidden_layers=12,
            hidden_size=embed_dim,
            num_attention_heads=12,
            intermediate_size=embed_dim * 4,  # Common practice is to have 4x the hidden_size
            hidden_act="gelu",
            layer_norm_eps=1e-12
        )

        self.vit = ViTModel(self.config)
        self.dist_token = DistillationToken(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dist_classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        embeddings = outputs.last_hidden_state
        
        
        embeddings = self.dist_token(embeddings)
        class_logits = self.classifier(embeddings[:, 0])
        dist_logits = self.dist_classifier(embeddings[:, 1])

        return class_logits, dist_logits
