import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
from src.models.vae import get_vae_model
from src.train_diffusion import MedMNISTProcessed
from src.utils.logger import setup_logger
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Vì VGG nhận 3 kênh, ta nhân bản kênh grayscale
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        h1_x = self.slice1(x)
        h1_y = self.slice1(y)
        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)
        h3_x = self.slice3(h2_x)
        h3_y = self.slice3(h2_y)
        return F.mse_loss(h1_x, h1_y) + F.mse_loss(h2_x, h2_y) + F.mse_loss(h3_x, h3_y)

def train_vae():
    # Config
    lr = 1e-4
    num_epochs = 100
    batch_size = 16
    img_size = 128
    kl_weight = 1e-6 # Trọng số cho KL Divergence để không làm át MSE
    perceptual_weight = 0.5 # Trọng số cho độ sắc nét
    output_dir = "artifacts/vae_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator = Accelerator(mixed_precision="fp16", log_with="tensorboard", project_dir="logs")
    logger = setup_logger('vae_train_logger', 'logs/train_vae.log')
    
    if accelerator.is_main_process:
        accelerator.init_trackers("vae_medmnist")

    # Model
    model = get_vae_model()
    perceptual_fn = PerceptualLoss().to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MedMNISTProcessed('data/processed/train_10.npz', transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"VAE Epoch {epoch}")
        
        for step, (batch_images, _) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Reconstruction
                reconstructed, posterior = model(batch_images)
                
                # 1. MSE Loss (Duy trì độ sáng/pixel)
                mse_loss = F.mse_loss(reconstructed, batch_images)
                
                # 2. Perceptual Loss (Tạo độ sắc nét giải phẫu)
                p_loss = perceptual_fn(reconstructed, batch_images)
                
                # 3. KL Loss (Phần cấu trúc latent)
                kl_loss = posterior.kl().mean()
                
                # Total Loss
                loss = mse_loss + perceptual_weight * p_loss + kl_weight * kl_loss
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            global_step += 1
            accelerator.log({
                "total_loss": loss.detach().item(),
                "mse_loss": mse_loss.detach().item(),
                "p_loss": p_loss.detach().item(),
                "kl_loss": kl_loss.detach().item()
            }, step=global_step)
            
        # Lưu mẫu để kiểm tra chất lượng tái tạo
        if (epoch + 1) % 20 == 0:
            if accelerator.is_main_process:
                # So sánh ảnh gốc và ảnh tái tạo
                comparison = torch.cat([batch_images[:4], reconstructed[:4]])
                utils.save_image(comparison, f"{output_dir}/recon_epoch_{epoch}.png", nrow=4, normalize=True)
                logger.info(f"Epoch {epoch} | Loss: {loss.item():.6f} | MSE: {mse_loss.item():.6f}")
                
                # Save checkpoint
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "best_vae.pt"))

    accelerator.end_training()
    logger.info("VAE Training complete. Model saved to artifacts/vae_v1/best_vae.pt")

if __name__ == "__main__":
    train_vae()
