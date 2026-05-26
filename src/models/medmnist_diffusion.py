import torch
from huggingface_hub import hf_hub_download
from typing import Optional
from src.models.vae import get_vae_model
from src.models.latent_unet import LatentConditionalUNet, get_latent_scheduler

class MedMNISTDiffusion:
    """
    End-to-end wrapper for generation using Latent Diffusion Models (LDM) on MedMNIST.
    Combines VAE and Latent UNet.
    """
    def __init__(
        self, 
        repo_id: Optional[str] = None, 
        local_vae_path: Optional[str] = None, 
        local_ldm_path: Optional[str] = None, 
        device: Optional[torch.device] = None
    ) -> None:
        self.device: torch.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate model components
        self.vae = get_vae_model().to(self.device)
        self.ldm = LatentConditionalUNet().to(self.device)
        self.scheduler = get_latent_scheduler()
        
        # Load weights
        if repo_id:
            print(f"Fetching models from Hugging Face: {repo_id}")
            vae_weights_path = hf_hub_download(repo_id=repo_id, filename="best_vae.pt")
            ldm_weights_path = hf_hub_download(repo_id=repo_id, filename="final_ldm.pt")
        else:
            if local_vae_path is None or local_ldm_path is None:
                raise ValueError("Must provide either repo_id or both local_vae_path and local_ldm_path.")
            vae_weights_path = local_vae_path
            ldm_weights_path = local_ldm_path
            
        self.vae.load_state_dict(torch.load(vae_weights_path, map_location=self.device))
        self.ldm.load_state_dict(torch.load(ldm_weights_path, map_location=self.device))
        
        self.vae.eval()
        self.ldm.eval()

    @torch.no_grad()
    def generate(
        self, 
        target_label: int, 
        num_samples: int = 1, 
        guidance_scale: float = 5.0, 
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate images conditioned on a specific class label using Classifier-Free Guidance (CFG).
        
        Args:
            target_label (int): Class label (e.g., 0 for Normal, 1 for Pneumonia).
            num_samples (int): Number of images to generate.
            guidance_scale (float): Strength of conditional guidance.
            num_inference_steps (int): Number of denoising steps.
            
        Returns:
            torch.Tensor: Generated images, shape (num_samples, 1, 128, 128) in range [0, 1].
        """
        # A. Sampling in Latent Space
        latents = torch.randn(num_samples, 4, 16, 16).to(self.device)
        conditional_labels = torch.full((num_samples,), target_label, dtype=torch.long).to(self.device)
        unconditional_labels = torch.full((num_samples,), 2, dtype=torch.long).to(self.device)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        for timestep in self.scheduler.timesteps:
            # Duplicate latents for CFG (unconditional and conditional)
            latent_model_input = torch.cat([latents, latents])
            labels_input = torch.cat([conditional_labels, unconditional_labels])
            
            # Predict noise
            noise_predictions = self.ldm(latent_model_input, timestep, labels_input)
            noise_pred_conditional, noise_pred_unconditional = noise_predictions.chunk(2)
            
            # Apply Classifier-Free Guidance
            adjusted_noise_pred = noise_pred_unconditional + guidance_scale * (noise_pred_conditional - noise_pred_unconditional)
            latents = self.scheduler.step(adjusted_noise_pred, timestep, latents).prev_sample
            
        # B. Decode latents to pixel space
        decoded_images = self.vae.vae.decode(latents).sample
        normalized_images = (decoded_images / 2 + 0.5).clamp(0, 1)
        
        return normalized_images
