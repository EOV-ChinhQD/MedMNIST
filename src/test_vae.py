import torch
import numpy as np
import os
from src.models.vae import get_vae_model
from src.train_diffusion import MedMNISTProcessed
from torchvision import transforms, utils
from PIL import Image

def test_vae(model_path='artifacts/vae_v2/best_vae.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "reports/vae_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Model
    model = get_vae_model().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Data Preparation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MedMNISTProcessed('data/processed/test.npz', transform=transform)
    
    # --- TEST 1: Reconstruction Test ---
    # Lấy 1 ca Normal (label 0) và 1 ca Pneumonia (label 1)
    idx_normal = np.where(dataset.labels == 0)[0][0]
    idx_pneumonia = np.where(dataset.labels == 1)[0][0]
    
    img_normal = dataset[idx_normal][0].unsqueeze(0).to(device)
    img_pneumonia = dataset[idx_pneumonia][0].unsqueeze(0).to(device)
    
    with torch.no_grad():
        recon_normal, _ = model(img_normal)
        recon_pneumonia, _ = model(img_pneumonia)
        
    # Lưu kết quả so sánh
    comparison = torch.cat([img_normal, recon_normal, img_pneumonia, recon_pneumonia])
    utils.save_image(comparison, f"{output_dir}/reconstruction_test.png", nrow=2, normalize=True)
    print(f"Reconstruction test saved to {output_dir}/reconstruction_test.png")
    
    # --- TEST 2: Latent Interpolation ---
    # Mã hóa 2 ca thành latent
    with torch.no_grad():
        z0 = model.vae.encode(img_normal).latent_dist.mean
        z1 = model.vae.encode(img_pneumonia).latent_dist.mean
        
        # Interpolation: 8 bước từ Normal sang Pneumonia
        interpolation_steps = []
        for alpha in np.linspace(0, 1, 8):
            z_interp = (1 - alpha) * z0 + alpha * z1
            img_interp = model.vae.decode(z_interp).sample
            interpolation_steps.append(img_interp)
            
        interpolation_result = torch.cat(interpolation_steps)
        utils.save_image(interpolation_result, f"{output_dir}/interpolation_test.png", nrow=8, normalize=True)
    
    print(f"Latent interpolation test saved to {output_dir}/interpolation_test.png")

if __name__ == "__main__":
    test_vae()
