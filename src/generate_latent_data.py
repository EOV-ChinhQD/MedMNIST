import torch
import numpy as np
import os
from src.models.vae import get_vae_model
from src.models.latent_unet import LatentConditionalUNet, get_latent_scheduler
from PIL import Image
from tqdm.auto import tqdm

def generate_latent_diffusion_data(vae_path, ldm_path, output_npz, num_samples_per_class=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load VAE (Decoder)
    vae = get_vae_model().to(device)
    vae.load_state_dict(torch.load(vae_path))
    vae.eval()
    
    # 2. Load Latent UNet
    ldm = LatentConditionalUNet().to(device)
    ldm.load_state_dict(torch.load(ldm_path))
    ldm.eval()
    
    scheduler = get_latent_scheduler()
    
    all_images = []
    all_labels = []
    
    print(f"Generating synthetic data via Latent Diffusion...")
    for label in [0, 1]:
        print(f"Generating class {label} ({'Normal' if label==0 else 'Pneumonia'})...")
        for i in tqdm(range(0, num_samples_per_class, 20)): # Batch size 20
            batch_size = min(20, num_samples_per_class - i)
            
            # A. Sampling in Latent Space (4, 16, 16)
            latents = torch.randn(batch_size, 4, 16, 16).to(device)
            labels = torch.full((batch_size,), label, dtype=torch.long).to(device)
            uncond_labels = torch.full((batch_size,), 2, dtype=torch.long).to(device) # Nhãn '2' là null
            
            guidance_scale = 5.0 # Tỉ lệ hướng dẫn (Gợi ý: 3.0 - 7.5)
            
            scheduler.set_timesteps(50)
            for t in scheduler.timesteps:
                with torch.no_grad():
                    # Dự đoán có nhãn và không nhãn để thực hiện CFG
                    # Ghép đôi batch để tính toán song song
                    latent_model_input = torch.cat([latents] * 2)
                    labels_input = torch.cat([labels, uncond_labels])
                    
                    noise_pred = ldm(latent_model_input, t, labels_input)
                    
                    # Tách kết quả
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    
                    # Công thức CFG: Đẩy mạnh đặc trưng của nhãn điều kiện
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # B. Giải nén Latent thành Pixels bằng VAE Decoder
            with torch.no_grad():
                images = vae.vae.decode(latents).sample
                
            # C. Hậu xử lý
            images = (images / 2 + 0.5).clamp(0, 1)
            images = (images.cpu().numpy() * 255).astype(np.uint8)
            images = images.squeeze(1) # (batch, 128, 128)
            
            all_images.append(images)
            all_labels.extend([label] * batch_size)
            
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.array(all_labels)
    
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, images=all_images, labels=all_labels)
    print(f"Success! Generated {len(all_images)} images saved to {output_npz}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, default="artifacts/vae_v2/best_vae.pt")
    parser.add_argument("--ldm", type=str, default="artifacts/latent_diffusion_v1/final_ldm.pt")
    parser.add_argument("--output", type=str, default="data/synthetic/latent_diffusion_data.npz")
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    
    generate_latent_diffusion_data(args.vae, args.ldm, args.output, num_samples_per_class=args.n)
