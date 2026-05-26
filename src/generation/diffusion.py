import torch
from src.models.diffusion import ConditionalDiffusionModel, get_scheduler
import numpy as np
import os
from typing import List

def generate_synthetic_data(model_path: str, output_npz: str, num_samples_per_class: int = 500, img_size: int = 128) -> None:
    """
    Generates synthetic data using a trained Conditional Diffusion Model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ConditionalDiffusionModel(img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scheduler = get_scheduler()
    
    all_images: List[np.ndarray] = []
    all_labels: List[int] = []
    
    for label in [0, 1]:
        print(f"Generating class {label}...")
        import math
        # Simple progress tracking
        for i in range(0, num_samples_per_class, 10): # Batch size 10
            batch_size: int = min(10, num_samples_per_class - i)
            
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
            samples_clamped = (samples / 2 + 0.5).clamp(0, 1)
            samples_np: np.ndarray = (samples_clamped.cpu().numpy() * 255).astype(np.uint8)
            samples_np = samples_np.squeeze(1) # (batch, h, w)
            
            all_images.append(samples_np)
            all_labels.extend([label] * batch_size)
            
    all_images_np: np.ndarray = np.concatenate(all_images, axis=0)
    all_labels_np: np.ndarray = np.array(all_labels)
    
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, images=all_images_np, labels=all_labels_np)
    print(f"Saved {len(all_images_np)} synthetic images to {output_npz}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples per class")
    args = parser.parse_args()
    
    generate_synthetic_data(args.model, args.output, num_samples_per_class=args.n)
