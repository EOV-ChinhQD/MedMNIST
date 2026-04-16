import torch
from src.models.gan import Generator
import numpy as np
import os
import argparse

def generate_gan_data(model_path, output_npz, num_samples_per_class=500, latent_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(latent_dim=latent_dim).to(device)
    netG.load_state_dict(torch.load(model_path))
    netG.eval()
    
    all_images = []
    all_labels = []

    for label in [0, 1]:
        print(f"Generating GAN class {label}...")
        with torch.no_grad():
            noise = torch.randn(num_samples_per_class, latent_dim, 1, 1, device=device)
            labels = torch.full((num_samples_per_class,), label, dtype=torch.long).to(device)
            fake = netG(noise, labels)
            
            fake = (fake / 2 + 0.5).clamp(0, 1)
            fake = (fake.cpu().numpy() * 255).astype(np.uint8)
            fake = fake.squeeze(1)
            
            all_images.append(fake)
            all_labels.extend([label] * num_samples_per_class)
            
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.array(all_labels)
    
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, images=all_images, labels=all_labels)
    print(f"Saved {len(all_images)} GAN synthetic images to {output_npz}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    
    generate_gan_data(args.model, args.output, num_samples_per_class=args.n)
