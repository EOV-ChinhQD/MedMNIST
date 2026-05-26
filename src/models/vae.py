import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from typing import Tuple

class MedMNISTVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for MedMNIST images, compressing 128x128 images to 16x16 latents.
    """
    def __init__(self, in_channels: int = 1, latent_channels: int = 4) -> None:
        super().__init__()
        self.vae = AutoencoderKL(
            in_channels=in_channels,
            out_channels=in_channels,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(64, 128, 256, 512),
            layers_per_block=2,
            latent_channels=latent_channels,
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        """
        Forward pass compressing images to latent space and then decoding back to pixel space.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, in_channels, 128, 128).
            
        Returns:
            Tuple[torch.Tensor, Distribution]: Decoded images and the latent posterior distribution.
        """
        posterior = self.vae.encode(images).latent_dist
        sampled_latents = posterior.sample()
        decoded_images = self.vae.decode(sampled_latents).sample
        return decoded_images, posterior

def get_vae_model() -> MedMNISTVAE:
    """
    Instantiates the MedMNISTVAE model.
    
    Returns:
        MedMNISTVAE: VAE model instance.
    """
    return MedMNISTVAE()
