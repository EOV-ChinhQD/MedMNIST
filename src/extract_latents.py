import torch
import numpy as np
import os
from src.models.vae import get_vae_model
from src.train_diffusion import MedMNISTProcessed
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

def extract_latents(batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_path = 'artifacts/vae_v2/best_vae.pt'
    output_path = 'data/processed/train_10_latents.npz'
    
    # 1. Load VAE
    model = get_vae_model().to(device)
    model.load_state_dict(torch.load(vae_path))
    model.eval()
    
    # 2. Load Data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MedMNISTProcessed('data/processed/train_10.npz', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_latents = []
    all_labels = []
    
    print("Extracting latents using VAE v2...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            # Encode ảnh thành phân phối latent và lấy giá trị trung bình (mean)
            # Dùng mean giúp dữ liệu huấn luyện Diffusion ổn định hơn
            latents = model.vae.encode(images).latent_dist.mean
            
            # Chuyển về CPU numpy để lưu
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Hợp nhất và lưu trữ
    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, latents=all_latents, labels=all_labels)
    
    print(f"Success! Latent dataset saved: {output_path}")
    print(f"Latent shape: {all_latents.shape}") # Kỳ vọng: (470, 4, 16, 16)

if __name__ == "__main__":
    extract_latents()
