import torch
from src.models.gan import Generator
import numpy as np
import os
import argparse
from typing import List

def generate_gan_data(model_path: str, output_npz: str, num_samples_per_class: int = 500, latent_dim: int = 100) -> None:
    """
    Generates synthetic data using a trained GAN Generator.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(latent_dim=latent_dim).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    all_images: List[np.ndarray] = []
    all_labels: List[int] = []

    for label in [0, 1]:
        print(f"Generating GAN class {label}...")
        with torch.no_grad():
            noise = torch.randn(num_samples_per_class, latent_dim, 1, 1, device=device)
            labels = torch.full((num_samples_per_class,), label, dtype=torch.long).to(device)
            fake = netG(noise, labels)
            
            fake_clamped = (fake / 2 + 0.5).clamp(0, 1)
            fake_np: np.ndarray = (fake_clamped.cpu().numpy() * 255).astype(np.uint8)
            fake_np = fake_np.squeeze(1)
            
            all_images.append(fake_np)
            all_labels.extend([label] * num_samples_per_class)
            
    all_images_np: np.ndarray = np.concatenate(all_images, axis=0)
    all_labels_np: np.ndarray = np.array(all_labels)
    
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, images=all_images_np, labels=all_labels_np)
    print(f"Saved {len(all_images_np)} GAN synthetic images to {output_npz}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained GAN Generator")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    parser.add_argument("--n", type=int, default=500, help="Number of samples per class")
    args = parser.parse_args()
    
    generate_gan_data(args.model, args.output, num_samples_per_class=args.n)
