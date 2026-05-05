import torch
from src.models.uncond_model import get_uncond_model, get_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_uncond_samples(model_path, output_img, img_size=128, n_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_uncond_model(img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    scheduler = get_scheduler()
    scheduler.set_timesteps(50)
    
    samples = torch.randn(n_samples, 1, img_size, img_size).to(device)
    
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(samples, t).sample
            samples = scheduler.step(noise_pred, t, samples).prev_sample
            
    samples = (samples / 2 + 0.5).clamp(0, 1).cpu().numpy().squeeze(1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(n_samples):
        ax = axes[i//5, i%5]
        ax.imshow(samples[i], cmap='gray')
        ax.set_title("Unconditional")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Saved samples to {output_img}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    generate_uncond_samples(args.model, args.output)
