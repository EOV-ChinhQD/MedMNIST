import medmnist
from medmnist import INFO, PneumoniaMNIST
import numpy as np
import os

def check_data():
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    print(f"Dataset: {info['python_class']}")
    print(f"Task: {info['task']}")
    print(f"Labels: {info['label']}")
    
    # Download 128x128
    train_dataset = PneumoniaMNIST(split='train', download=True, size=128)
    val_dataset = PneumoniaMNIST(split='val', download=True, size=128)
    test_dataset = PneumoniaMNIST(split='test', download=True, size=128)
    
    print(f"Train samples: {len(train_dataset)}")
    
    # Check class distribution
    labels = train_dataset.labels.flatten()
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Train class distribution: {dist}")
    
    # 10% split
    num_samples = len(train_dataset)
    num_10 = int(0.1 * num_samples)
    print(f"10% sample size: {num_10}")

if __name__ == "__main__":
    check_data()
