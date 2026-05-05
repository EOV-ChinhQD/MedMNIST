import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import numpy as np
import os
import argparse
from src.models.classifier import get_resnet18
from src.utils.logger import setup_logger
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import medmnist
from medmnist import Evaluator

class NPZDataset(Dataset):
    def __init__(self, npz_path, transform=None, images=None, labels=None):
        if images is not None and labels is not None:
            self.images = images
            self.labels = labels.flatten()
        else:
            data = np.load(npz_path)
            self.images = data['images']
            self.labels = data['labels'].flatten()
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)

def train_classifier(args):
    logger = setup_logger(args.name, f'logs/{args.name}.log')
    logger.info(f"Training scenario: {args.name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Datasets & Transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    if args.aug == 'traditional':
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load primary data
    train_dataset = NPZDataset(args.train_path, transform=train_transform)
    
    # If synthetic data provided, combine
    if args.synthetic_path:
        logger.info(f"Loading synthetic data from {args.synthetic_path}")
        synth_data = np.load(args.synthetic_path)
        synth_dataset = NPZDataset(None, transform=train_transform, images=synth_data['images'], labels=synth_data['labels'])
        train_dataset = ConcatDataset([train_dataset, synth_dataset])
        logger.info(f"Combined dataset size: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = NPZDataset('data/processed/val.npz', transform=val_test_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    test_dataset = NPZDataset('data/processed/test.npz', transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Model, Loss, Optimizer
    model = get_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 3. Training Loop
    best_val_auc = 0
    os.makedirs('artifacts/classifiers', exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(lbls.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'artifacts/classifiers/{args.name}_best.pt')
            
    # 4. Final Evaluation on Test Set
    logger.info("Final Evaluation on Test Set")
    model.load_state_dict(torch.load(f'artifacts/classifiers/{args.name}_best.pt'))
    model.eval()
    test_probs = []
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(lbls.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)
    auc = roc_auc_score(test_labels, test_probs)
    
    logger.info(f"RES: {args.name} | ACC: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    # Save results
    with open(f'reports/results_{args.name}.txt', 'w') as f:
        f.write(f"Scenario: {args.name}\n")
        f.write(f"ACC: {acc:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--train_path", type=str, default='data/processed/train_10.npz')
    parser.add_argument("--synthetic_path", type=str, default=None)
    parser.add_argument("--aug", type=str, default='none')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train_classifier(args)
