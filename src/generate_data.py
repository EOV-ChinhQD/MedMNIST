import torch
from src.models.diffusion_model import ConditionalDiffusionModel, get_scheduler
import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm

def generate_synthetic_data(model_path, output_npz, num_samples_per_class=500, img_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ConditionalDiffusionModel(img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    scheduler = get_scheduler()
    
    all_images = []
    all_labels = []
    
    for label in [0, 1]:
        print(f"Generating class {label}...")
        for i in tqdm(range(0, num_samples_per_class, 10)): # Batch size 10
            batch_size = min(10, num_samples_per_class - i)
            
            # Start from pure noise
            samples = torch.randn(batch_size, 1, img_size, img_size).to(device)
            labels = torch.full((batch_size,), label, dtype=torch.long).to(device)
            
            # Reverse Diffusion
            scheduler.set_timesteps(50) # Use 50 steps for inference to save time
            for t in scheduler.timesteps:
                with torch.no_grad():
                    # Predict noise
                    noise_pred = model(samples, t, labels)
                    
                    # Compute previous image
                    samples = scheduler.step(noise_pred, t, samples).prev_sample
            
            # Post-process
            samples = (samples / 2 + 0.5).clamp(0, 1)
            samples = (samples.cpu().numpy() * 255).astype(np.uint8)
            samples = samples.squeeze(1) # (batch, h, w)
            
            all_images.append(samples)
            all_labels.extend([label] * batch_size)
            
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.array(all_labels)
    
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, images=all_images, labels=all_labels)
    print(f"Saved {len(all_images)} synthetic images to {output_npz}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n", type=int, default=1000) # samples per class
    args = parser.parse_args()
    
    generate_synthetic_data(args.model, args.output, num_samples_per_class=args.n)
