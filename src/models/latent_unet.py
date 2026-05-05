import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler

class LatentConditionalUNet(nn.Module):
    def __init__(self, num_classes=3, latent_channels=4, embedding_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        self.embedding_dim = embedding_dim
        
        # Nhãn nhúng (Label Embedding)
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        
        # UNet làm việc trên không gian latent 16x16
        self.unet = UNet2DConditionModel(
            sample_size=16,
            in_channels=latent_channels, 
            out_channels=latent_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512), # Cấu trúc tinh gọn cho 16x16
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D", # Cross-Attention cho nhãn
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=embedding_dim,
        )
        
    def forward(self, latent_input, timesteps, labels):
        # latent_input: (batch, 4, 16, 16)
        # labels: (batch,)
        
        # Chuyển nhãn thành embedding và đưa vào Cross-Attention
        embeddings = self.label_emb(labels).unsqueeze(1) # (batch, 1, 256)
        
        # Dự đoán nhiễu trong không gian latent
        noise_pred = self.unet(latent_input, timesteps, encoder_hidden_states=embeddings).sample
        return noise_pred

def get_latent_scheduler():
    return DDPMScheduler(num_train_timesteps=1000)
