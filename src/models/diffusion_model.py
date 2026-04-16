import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, num_classes=2, img_size=128, embedding_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        
        # Label embedding: map label to a vector that can be used as 'encoder_hidden_states'
        # UNet2DConditionModel expects hidden_states of shape (batch, sequence_length, embedding_dim)
        # We can treat each label as a sequence of length 1.
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        
        self.unet = UNet2DConditionModel(
            sample_size=img_size,
            in_channels=1, # Grayscale
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=embedding_dim,
        )
        
    def forward(self, x, timesteps, labels):
        # x: (batch, 1, h, w)
        # timesteps: (batch,)
        # labels: (batch,)
        
        # Get embeddings: (batch, embedding_dim)
        embeddings = self.label_emb(labels)
        
        # Reshape for cross-attention: (batch, 1, embedding_dim)
        embeddings = embeddings.unsqueeze(1)
        
        # Predict noise
        noise_pred = self.unet(x, timesteps, encoder_hidden_states=embeddings).sample
        return noise_pred

def get_scheduler():
    return DDPMScheduler(num_train_timesteps=1000)
