import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler

def get_uncond_model(img_size: int = 128) -> UNet2DModel:
    """
    Instantiates an unconditional UNet model for image generation.
    
    Args:
        img_size (int): Expected spatial size of input images.
        
    Returns:
        UNet2DModel: Unconditional UNet.
    """
    model = UNet2DModel(
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
            "DownBlock2D",  # No Cross-Attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

def get_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    """
    Instantiates a DDPMScheduler.
    
    Args:
        num_train_timesteps (int): Total number of timesteps for the diffusion process.
        
    Returns:
        DDPMScheduler: Configured scheduler.
    """
    return DDPMScheduler(num_train_timesteps=num_train_timesteps)
