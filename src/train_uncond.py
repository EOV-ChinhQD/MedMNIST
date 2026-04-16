import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
from src.models.uncond_model import get_uncond_model, get_scheduler
from src.train_diffusion import MedMNISTProcessed # reuse
from src.utils.logger import setup_logger

def train_uncond():
    # Config
    lr = 1e-4
    num_epochs = 50 # Running fast for baseline comparison
    batch_size = 16
    img_size = 128
    output_dir = "artifacts/uncond_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator = Accelerator(mixed_precision="fp16")
    logger = setup_logger('uncond_logger', 'logs/train_uncond.log')
    
    model = get_uncond_model(img_size=img_size)
    noise_scheduler = get_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MedMNISTProcessed('data/processed/train_10.npz', transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_images, _ in train_dataloader:
            noise = torch.randn(batch_images.shape).to(batch_images.device)
            bs = batch_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=batch_images.device).long()
            noisy_images = noise_scheduler.add_noise(batch_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch} Loss: {loss.item()}")

    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), f"{output_dir}/uncond_final.pt")
        logger.info("Unconditional training finished.")

if __name__ == "__main__":
    train_uncond()
