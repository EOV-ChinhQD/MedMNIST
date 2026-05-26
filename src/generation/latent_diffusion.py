import torch
import numpy as np
import os
from src.models.vae import get_vae_model
from src.models.latent_unet import LatentConditionalUNet, get_latent_scheduler
from typing import List

def generate_latent_diffusion_data(vae_path: str, ldm_path: str, output_npz: str, num_samples_per_class: int = 1000) -> None:
    """
    Generates synthetic data using a trained Latent Diffusion Model (LDM) and VAE Decoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load VAE (Decoder)
    vae = get_vae_model().to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    
    # 2. Load Latent UNet
    ldm = LatentConditionalUNet().to(device)
    ldm.load_state_dict(torch.load(ldm_path, map_location=device))
    ldm.eval()
    
    scheduler = get_latent_scheduler()
    
    all_images: List[np.ndarray] = []
    all_labels: List[int] = []
    
    print(f"Generating synthetic data via Latent Diffusion...")
    for label in [0, 1]:
        print(f"Generating class {label} ({'Normal' if label==0 else 'Pneumonia'})...")
        for i in range(0, num_samples_per_class, 20): # Batch size 20
            batch_size: int = min(20, num_samples_per_class - i)
            
            # A. Sampling in Latent Space (4, 16, 16)
            latents = torch.randn(batch_size, 4, 16, 16).to(device)
            labels = torch.full((batch_size,), label, dtype=torch.long).to(device)
            uncond_labels = torch.full((batch_size,), 2, dtype=torch.long).to(device) # '2' is null label
            
            guidance_scale: float = 5.0 # CFG Scale
            
            scheduler.set_timesteps(50)
            for t in scheduler.timesteps:
                with torch.no_grad():
                    # Predict conditional and unconditional noise for Classifier-Free Guidance
                    latent_model_input = torch.cat([latents] * 2)
                    labels_input = torch.cat([labels, uncond_labels])
                    
                    noise_pred = ldm(latent_model_input, t, labels_input)
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    
                    # CFG Formula
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # B. Decode Latents to Pixels using VAE Decoder
            with torch.no_grad():
                images_tensor = vae.vae.decode(latents).sample
                
            # C. Post-process
            images_clamped = (images_tensor / 2 + 0.5).clamp(0, 1)
            images_np: np.ndarray = (images_clamped.cpu().numpy() * 255).astype(np.uint8)
            images_np = images_np.squeeze(1) # (batch, 128, 128)
            
            all_images.append(images_np)
            all_labels.extend([label] * batch_size)
            
    all_images_np: np.ndarray = np.concatenate(all_images, axis=0)
    all_labels_np: np.ndarray = np.array(all_labels)
    
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, images=all_images_np, labels=all_labels_np)
    print(f"Success! Generated {len(all_images_np)} images saved to {output_npz}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, default="artifacts/vae_v2/best_vae.pt")
    parser.add_argument("--ldm", type=str, default="artifacts/latent_diffusion_v1/final_ldm.pt")
    parser.add_argument("--output", type=str, default="data/synthetic/latent_diffusion_data.npz")
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    
    generate_latent_diffusion_data(args.vae, args.ldm, args.output, num_samples_per_class=args.n)
