import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler

class ConditionalDiffusionModel(nn.Module):
    """
    Conditional Diffusion Model for image generation based on class labels.
    """
    def __init__(self, num_classes: int = 2, img_size: int = 128, embedding_dim: int = 256) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.img_size: int = img_size
        self.embedding_dim: int = embedding_dim
        
        # Label embedding maps discrete labels to continuous vectors
        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)
        
        self.unet = UNet2DConditionModel(
            sample_size=img_size,
            in_channels=1,
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
        
    def forward(self, images: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the diffusion model.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 1, height, width).
            timesteps (torch.Tensor): Noise timesteps of shape (batch_size,).
            labels (torch.Tensor): Class labels of shape (batch_size,).
            
        Returns:
            torch.Tensor: Predicted noise.
        """
        # Obtain embeddings and reshape for cross-attention
        label_embeddings = self.label_embedding(labels).unsqueeze(1)
        
        # Predict noise using UNet
        noise_prediction = self.unet(
            images, 
            timesteps, 
            encoder_hidden_states=label_embeddings
        ).sample
        return noise_prediction

def get_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    """
    Instantiates a DDPMScheduler.
    
    Args:
        num_train_timesteps (int): Total number of timesteps for the diffusion process.
        
    Returns:
        DDPMScheduler: Configured scheduler.
    """
    return DDPMScheduler(num_train_timesteps=num_train_timesteps)
