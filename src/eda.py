import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.logger import setup_logger

def run_eda():
    logger = setup_logger('eda_logger', 'logs/eda.log')
    logger.info("Starting EDA")
    
    data = np.load('data/processed/train_10.npz')
    images = data['images']
    labels = data['labels']
    
    logger.info(f"Loaded {len(images)} samples from train_10")
    
    # 1. Visualization
    classes = {0: 'Normal', 1: 'Pneumonia'}
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(2):
        class_indices = np.where(labels == i)[0]
        selected = np.random.choice(class_indices, 5, replace=False)
        for j, idx in enumerate(selected):
            axes[i, j].imshow(images[idx], cmap='gray')
            axes[i, j].set_title(classes[i])
            axes[i, j].axis('off')
    
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/sample_images.png')
    logger.info("Saved sample images to reports/figures/sample_images.png")
    
    # 2. Stats
    mean = images.mean() / 255.0
    std = images.std() / 255.0
    logger.info(f"Dataset stats: Mean={mean:.4f}, Std={std:.4f}")
    
    with open('reports/eda_summary.md', 'w') as f:
        f.write("# EDA Summary for PneumoniaMNIST (Scarcity Split)\n\n")
        f.write(f"- Total Samples (10%): {len(images)}\n")
        f.write(f"- Class Distribution: Normal: {np.sum(labels==0)}, Pneumonia: {np.sum(labels==1)}\n")
        f.write(f"- Image Stats: Mean={mean:.4f}, Std={std:.4f}\n")
        f.write(f"- Resolution: {images.shape[1]}x{images.shape[2]}\n")
        f.write("\n![Samples](figures/sample_images.png)\n")

if __name__ == "__main__":
    run_eda()
