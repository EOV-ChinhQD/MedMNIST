import torch
from src.models.uncond import get_uncond_model, get_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import List

def generate_uncond_samples(model_path: str, output_img: str, img_size: int = 128, n_samples: int = 10) -> None:
    """
    Generates a few samples and saves them to an image grid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_uncond_model(img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scheduler = get_scheduler()
    scheduler.set_timesteps(50)
    
    samples = torch.randn(n_samples, 1, img_size, img_size).to(device)
    
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(samples, t).sample
            samples = scheduler.step(noise_pred, t, samples).prev_sample
            
    samples_np: np.ndarray = (samples / 2 + 0.5).clamp(0, 1).cpu().numpy().squeeze(1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(n_samples):
        ax = axes[i//5, i%5]
        ax.imshow(samples_np[i], cmap='gray')
        ax.set_title("Unconditional")
        ax.axis('off')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img)
    print(f"Saved samples to {output_img}")

def generate_uncond_npz(model_path: str, output_npz: str, n_samples: int = 500, img_size: int = 128) -> None:
    """
    Generates a large number of samples and saves them to a .npz file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_uncond_model(img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scheduler = get_scheduler()
    scheduler.set_timesteps(50)
    
    all_images: List[np.ndarray] = []
    batch_size: int = 20
    
    for i in range(0, n_samples, batch_size):
        curr_bs: int = min(batch_size, n_samples - i)
        print(f"Generating batch {i//batch_size} ({curr_bs} samples)...")
        samples = torch.randn(curr_bs, 1, img_size, img_size).to(device)
        
        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(samples, t).sample
                samples = scheduler.step(noise_pred, t, samples).prev_sample
        
        imgs_np: np.ndarray = (samples / 2 + 0.5).clamp(0, 1).cpu().numpy().squeeze(1)
        all_images.append((imgs_np * 255).astype(np.uint8))
        
    all_images_np: np.ndarray = np.concatenate(all_images, axis=0)
    # Assign random labels 50/50 for unconditional baseline
    all_labels_np: np.ndarray = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, images=all_images_np, labels=all_labels_np)
    print(f"Saved {len(all_images_np)} Unconditional images to {output_npz}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, required=True, help="Output path (can be .png or .npz)")
    parser.add_argument("--n", type=int, default=500, help="Number of samples")
    parser.add_argument("--mode", type=str, choices=['img', 'npz'], default='npz')
    args = parser.parse_args()
    
    if args.mode == 'img':
        generate_uncond_samples(args.model, args.output, n_samples=args.n)
    else:
        generate_uncond_npz(args.model, args.output, n_samples=args.n)
