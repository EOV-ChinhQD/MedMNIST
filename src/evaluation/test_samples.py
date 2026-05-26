import torch
from src.models.diffusion import ConditionalDiffusionModel, get_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List

def test_samples(model_path: str, output_img: str, img_size: int = 128) -> None:
    """
    Generates samples using the trained Conditional Diffusion Model and saves them to an image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusionModel(img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scheduler = get_scheduler()
    scheduler.set_timesteps(50) # Fast sampling
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    classes: Dict[int, str] = {0: 'Normal', 1: 'Pneumonia'}
    
    for label in [0, 1]:
        print(f"Sampling class {label}...")
        samples = torch.randn(5, 1, img_size, img_size).to(device)
        cond_labels = torch.full((5,), label, dtype=torch.long).to(device)
        
        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(samples, t, cond_labels)
                samples = scheduler.step(noise_pred, t, samples).prev_sample
        
        # Scale back to [0, 1] for visualization
        samples_np: np.ndarray = (samples / 2 + 0.5).clamp(0, 1).cpu().numpy().squeeze(1)
        for j in range(5):
            axes[label, j].imshow(samples_np[j], cmap='gray')
            axes[label, j].set_title(classes[label])
            axes[label, j].axis('off')
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img)
    print(f"Saved test samples to {output_img}")

if __name__ == "__main__":
    import glob
    checkpoints: List[str] = sorted(glob.glob('artifacts/diffusion_v1/checkpoint_epoch_*.pt'))
    
    latest: str = ""
    if checkpoints:
        latest = checkpoints[-1]
    elif os.path.exists('artifacts/diffusion_v1/final_model.pt'):
        latest = 'artifacts/diffusion_v1/final_model.pt'
        
    if latest:
        print(f"Using model: {latest}")
        test_samples(latest, 'reports/figures/guided_samples_final.png')
    else:
        print("No models found.")
