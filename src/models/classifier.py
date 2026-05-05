import torch
import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=1, pretrained=True):
    # For binary classification on MedMNIST, usually we use 1 output with BCEWithLogitsLoss
    # or 2 outputs with CrossEntropyLoss. MedMNIST evaluators handle both.
    # We'll use 2 outputs for multi-class style (0: normal, 1: pneumonia)
    resnet = models.resnet18(pretrained=pretrained)
    
    # Change first layer for grayscale (1 channel)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Change last layer for num_classes
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    
    return resnet
