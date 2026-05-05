import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class MedMNISTVAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4):
        super().__init__()
        # Sử dụng AutoencoderKL từ diffusers làm kiến trúc chuẩn
        # Nén ảnh 128x128 -> Latent 16x16 (giảm 8 lần)
        self.vae = AutoencoderKL(
            in_channels=in_channels,
            out_channels=in_channels,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(64, 128, 256, 512),
            layers_per_block=2,
            latent_channels=latent_channels,
        )

    def forward(self, x):
        # x: (batch, 1, 128, 128)
        # posteriors là phân phối xác suất trong latent space
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample()
        dec = self.vae.decode(z).sample
        return dec, posterior

def get_vae_model():
    return MedMNISTVAE()
