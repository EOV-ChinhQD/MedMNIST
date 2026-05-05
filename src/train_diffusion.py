import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
from src.models.diffusion_model import ConditionalDiffusionModel, get_scheduler
from src.utils.logger import setup_logger
from PIL import Image

class MedMNISTProcessed(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data['images']
        self.labels = data['labels'].flatten()
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)

def train():
    # Config
    lr = 1e-4
    num_epochs = 100
    batch_size = 16
    img_size = 128
    output_dir = "artifacts/diffusion_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator = Accelerator(mixed_precision="fp16", log_with="tensorboard", project_dir="logs")
    logger = setup_logger('train_logger', 'logs/train_diffusion.log')
    
    if accelerator.is_main_process:
        accelerator.init_trackers("diffusion_pneumonia")

    # Model & Scheduler
    model = ConditionalDiffusionModel(img_size=img_size)
    noise_scheduler = get_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # Scale to [-1, 1]
    ])
    dataset = MedMNISTProcessed('data/processed/train_10.npz', transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, (batch_images, batch_labels) in enumerate(train_dataloader):
            # Sample noise
            noise = torch.randn(batch_images.shape).to(batch_images.device)
            bs = batch_images.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=batch_images.device).long()
            
            # Add noise to images
            noisy_images = noise_scheduler.add_noise(batch_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict noise
                noise_pred = model(noisy_images, timesteps, batch_labels)
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"loss": loss.detach().item()}, step=global_step)
            
        # Optional: Sampling at end of epoch
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch} finished. Loss: {loss.item()}")
            # Save checkpoint
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))
    
    accelerator.end_training()
    # Final save
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "final_model.pt"))
        logger.info("Training complete. Model saved.")

if __name__ == "__main__":
    train()
