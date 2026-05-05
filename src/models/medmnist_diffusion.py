import torch
import os
from huggingface_hub import hf_hub_download
from src.models.vae import get_vae_model
from src.models.latent_unet import LatentConditionalUNet, get_latent_scheduler
import numpy as np

class MedMNISTDiffusion:
    def __init__(self, repo_id=None, local_vae=None, local_ldm=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.vae = get_vae_model().to(self.device)
        self.ldm = LatentConditionalUNet().to(self.device)
        self.scheduler = get_latent_scheduler()
        
        if repo_id:
            print(f"Fetching models from Hugging Face: {repo_id}")
            vae_path = hf_hub_download(repo_id=repo_id, filename="best_vae.pt")
            ldm_path = hf_hub_download(repo_id=repo_id, filename="final_ldm.pt")
        else:
            vae_path = local_vae
            ldm_path = local_ldm
            
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        self.ldm.load_state_dict(torch.load(ldm_path, map_location=self.device))
        
        self.vae.eval()
        self.ldm.eval()

    @torch.no_grad()
    def generate(self, label, num_samples=1, guidance_scale=5.0, num_inference_steps=50):
        """
        label: 0 for Normal, 1 for Pneumonia
        """
        # A. Sampling in Latent Space
        latents = torch.randn(num_samples, 4, 16, 16).to(self.device)
        labels = torch.full((num_samples,), label, dtype=torch.long).to(self.device)
        uncond_labels = torch.full((num_samples,), 2, dtype=torch.long).to(self.device)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            labels_input = torch.cat([labels, uncond_labels])
            
            noise_pred = self.ldm(latent_model_input, t, labels_input)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            
            # CFG
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # B. Decode to Pixels
        images = self.vae.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return images # Tensor (N, 1, 128, 128)

# Example Usage:
# model = MedMNISTDiffusion(repo_id="Kenkaw303/medmnist-pneumonia-diffusion")
# img = model.generate(label=1)
