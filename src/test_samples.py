import torch
from src.models.diffusion_model import ConditionalDiffusionModel, get_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os

def test_samples(model_path, output_img, img_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusionModel(img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    scheduler = get_scheduler()
    scheduler.set_timesteps(50) # Fast sampling
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    classes = {0: 'Normal', 1: 'Pneumonia'}
    
    for label in [0, 1]:
        print(f"Sampling class {label}...")
        samples = torch.randn(5, 1, img_size, img_size).to(device)
        cond_labels = torch.full((5,), label, dtype=torch.long).to(device)
        
        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(samples, t, cond_labels)
                samples = scheduler.step(noise_pred, t, samples).prev_sample
        
        samples = (samples / 2 + 0.5).clamp(0, 1).cpu().numpy().squeeze(1)
        for j in range(5):
            axes[label, j].imshow(samples[j], cmap='gray')
            axes[label, j].set_title(classes[label])
            axes[label, j].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Saved test samples to {output_img}")

if __name__ == "__main__":
    import glob
    checkpoints = sorted(glob.glob('artifacts/diffusion_v1/checkpoint_epoch_*.pt'))
    if checkpoints:
        latest = checkpoints[-1]
    elif os.path.exists('artifacts/diffusion_v1/final_model.pt'):
        latest = 'artifacts/diffusion_v1/final_model.pt'
    else:
        latest = None
        
    if latest:
        print(f"Using model: {latest}")
        test_samples(latest, 'reports/figures/guided_samples_final.png')
    else:
        print("No models found.")
