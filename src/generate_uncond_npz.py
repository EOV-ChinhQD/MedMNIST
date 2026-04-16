import torch
from src.models.uncond_model import get_uncond_model, get_scheduler
import numpy as np
import os
import argparse

def generate_uncond_npz(model_path, output_npz, n_samples=500, img_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_uncond_model(img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    scheduler = get_scheduler()
    scheduler.set_timesteps(50)
    
    all_images = []
    # Generate in batches to avoid OOM
    batch_size = 20
    for i in range(0, n_samples, batch_size):
        curr_bs = min(batch_size, n_samples - i)
        print(f"Generating batch {i//batch_size} ({curr_bs} samples)...")
        samples = torch.randn(curr_bs, 1, img_size, img_size).to(device)
        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(samples, t).sample
                samples = scheduler.step(noise_pred, t, samples).prev_sample
        
        imgs = (samples / 2 + 0.5).clamp(0, 1).cpu().numpy().squeeze(1)
        all_images.append((imgs * 255).astype(np.uint8))
        
    all_images = np.concatenate(all_images, axis=0)
    # Assign random labels 50/50 for unconditional baseline
    all_labels = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, images=all_images, labels=all_labels)
    print(f"Saved {len(all_images)} Unconditional images to {output_npz}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    generate_uncond_npz(args.model, args.output, n_samples=args.n)
