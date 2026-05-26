import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet

def get_resnet18(num_classes: int = 2, pretrained: bool = True) -> ResNet:
    """
    Instantiates a ResNet18 model configured for grayscale images and a specific number of output classes.
    
    Args:
        num_classes (int): Number of output classes (e.g., 2 for normal vs pneumonia).
        pretrained (bool): Whether to load pre-trained weights from ImageNet.
        
    Returns:
        ResNet: Modified ResNet18 model.
    """
    resnet = models.resnet18(pretrained=pretrained)
    
    # Change first convolutional layer to accept grayscale images (1 channel)
    resnet.conv1 = nn.Conv2d(
        in_channels=1, 
        out_channels=64, 
        kernel_size=7, 
        stride=2, 
        padding=3, 
        bias=False
    )
    
    # Adjust final fully connected layer to match number of classes
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features=num_features, out_features=num_classes)
    
    return resnet
