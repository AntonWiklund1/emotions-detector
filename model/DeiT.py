import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
import constants

torch.manual_seed(42)

image_size = constants.image_size
patch_size = constants.patch_size
num_heads = constants.num_heads
num_layers = constants.num_layers

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
            num_hidden_layers=num_layers,  # Number of transformer layers
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            intermediate_size=embed_dim * 4,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            num_channels=1  # Grayscale images
        )

        # Load pre-trained ViT model
        self.vit = ViTModel(self.config)
        
        # Add distillation token
        self.dist_token = DistillationToken(embed_dim)
        
        # Add classification heads
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dist_classifier = nn.Linear(embed_dim, num_classes)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._initialize_weights()

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

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
