import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
from src.models.latent_unet import LatentConditionalUNet, get_latent_scheduler
from src.utils.logger import setup_logger
from typing import Tuple

class LatentDataset(Dataset):
    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        self.latents: np.ndarray = data['latents']
        self.labels: np.ndarray = data['labels'].flatten()
        
    def __len__(self) -> int:
        return len(self.latents)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.latents[idx]
        label = self.labels[idx]
        return torch.tensor(latent, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_latent_diffusion() -> None:
    """
    Trains the Latent Diffusion Model on extracted latents.
    """
    # Config
    lr: float = 1e-4
    num_epochs: int = 200 # Train longer for LDM
    batch_size: int = 32
    output_dir: str = "artifacts/latent_diffusion_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator = Accelerator(mixed_precision="fp16", log_with="tensorboard", project_dir="logs")
    logger = setup_logger('ldm_train_logger', 'logs/train_ldm.log')
    
    if accelerator.is_main_process:
        accelerator.init_trackers("latent_diffusion_medmnist")

    # Model & Scheduler
    model = LatentConditionalUNet()
    noise_scheduler = get_latent_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Data
    dataset = LatentDataset('data/processed/train_10_latents.npz')
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    global_step: int = 0
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"LDM Epoch {epoch}")
        
        for step, (latents, labels) in enumerate(train_dataloader):
            # 1. Sample noise (latent size 4x16x16)
            noise = torch.randn(latents.shape).to(latents.device)
            bs: int = latents.shape[0]
            
            # --- CFG: Label Dropping (10%) ---
            drop_mask = torch.rand(bs, device=latents.device) < 0.1
            labels = labels.clone()
            labels[drop_mask] = 2 # Null label
            # --------------------------------
            
            # 2. Sample random timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device).long()
            
            # 3. Add noise (Forward Diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with accelerator.accumulate(model):
                # 4. Predict noise (Conditioned on labels)
                noise_pred = model(noisy_latents, timesteps, labels)
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"ldm_loss": loss.detach().item()}, step=global_step)
            
        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch} | Loss: {loss.item():.6f}")
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "final_ldm.pt"))
        logger.info("Latent Diffusion training complete.")

if __name__ == "__main__":
    train_latent_diffusion()
