import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler

class LatentConditionalUNet(nn.Module):
    """
    Conditional UNet designed to operate in the latent space (e.g., 16x16 resolution).
    """
    def __init__(self, num_classes: int = 3, latent_channels: int = 4, embedding_dim: int = 256) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.latent_channels: int = latent_channels
        self.embedding_dim: int = embedding_dim
        
        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)
        
        self.unet = UNet2DConditionModel(
            sample_size=16,
            in_channels=latent_channels, 
            out_channels=latent_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=embedding_dim,
        )
        
    def forward(self, latent_inputs: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the latent conditional UNet.
        
        Args:
            latent_inputs (torch.Tensor): Latent representation, shape (batch_size, latent_channels, 16, 16).
            timesteps (torch.Tensor): Noise timesteps, shape (batch_size,).
            labels (torch.Tensor): Class labels, shape (batch_size,).
            
        Returns:
            torch.Tensor: Predicted noise in latent space.
        """
        embedded_labels = self.label_embedding(labels).unsqueeze(1)
        noise_prediction = self.unet(
            latent_inputs, 
            timesteps, 
            encoder_hidden_states=embedded_labels
        ).sample
        return noise_prediction

def get_latent_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    """
    Instantiates a DDPMScheduler for the latent UNet.
    
    Args:
        num_train_timesteps (int): Total number of timesteps for the diffusion process.
        
    Returns:
        DDPMScheduler: Configured scheduler.
    """
    return DDPMScheduler(num_train_timesteps=num_train_timesteps)
