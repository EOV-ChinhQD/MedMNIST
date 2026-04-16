import os
import numpy as np
import medmnist
from medmnist import PneumoniaMNIST
from sklearn.model_selection import train_test_split
from src.utils.logger import ETL_Logger
import torch

def prepare_data(size=128, scarcity_ratio=0.1):
    etl_logger = ETL_Logger()
    etl_logger.log_transformation(f"Starting ETL for PneumoniaMNIST size={size}")
    
    # Load all sets
    train_dataset = PneumoniaMNIST(split='train', download=True, size=size)
    val_dataset = PneumoniaMNIST(split='val', download=True, size=size)
    test_dataset = PneumoniaMNIST(split='test', download=True, size=size)
    
    etl_logger.log_metadata({
        "train_original_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset)
    })
    
    # Convert to numpy for slicing
    x_train = train_dataset.imgs
    y_train = train_dataset.labels.flatten()
    
    # Create stratified split for 10%
    x_10, _, y_10, _ = train_test_split(
        x_train, y_train, 
        train_size=scarcity_ratio, 
        stratify=y_train, 
        random_state=42
    )
    
    etl_logger.log_transformation(f"Created scarcity split: {len(x_10)} samples")
    
    # Quality Checks
    unique, counts = np.unique(y_10, return_counts=True)
    dist_10 = dict(zip(unique.astype(int).tolist(), counts.tolist()))
    etl_logger.log_quality_check("Split Balance", "Success", f"10% Dist: {dist_10}")
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    
    np.savez_compressed('data/processed/train_10.npz', images=x_10, labels=y_10)
    np.savez_compressed('data/processed/train_full.npz', images=x_train, labels=y_train)
    np.savez_compressed('data/processed/val.npz', images=val_dataset.imgs, labels=val_dataset.labels)
    np.savez_compressed('data/processed/test.npz', images=test_dataset.imgs, labels=test_dataset.labels)
    
    etl_logger.log_transformation("Data saved to data/processed/")
    
    # Verify no duplicates (simple check by sum if needed, but indices are better)
    # Since we used train_test_split, they are disjoint from the remaining 90%.
    
if __name__ == "__main__":
    prepare_data(size=128, scarcity_ratio=0.1)
