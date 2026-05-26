import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Conditional Generative Adversarial Network (GAN) Generator.
    """
    def __init__(self, latent_dim: int = 100, num_classes: int = 2, img_size: int = 128) -> None:
        super().__init__()
        self.latent_dim: int = latent_dim
        self.num_classes: int = num_classes
        self.img_size: int = img_size
        
        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)
        
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim + num_classes, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generating images from noise and labels.
        
        Args:
            noise (torch.Tensor): Latent noise vectors, shape (batch_size, latent_dim, 1, 1).
            labels (torch.Tensor): Class labels, shape (batch_size,).
            
        Returns:
            torch.Tensor: Generated images.
        """
        embedded_labels = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        combined_input = torch.cat([noise, embedded_labels], dim=1)
        return self.network(combined_input)

class Discriminator(nn.Module):
    """
    Conditional Generative Adversarial Network (GAN) Discriminator.
    """
    def __init__(self, num_classes: int = 2, img_size: int = 128) -> None:
        super().__init__()
        self.img_size: int = img_size
        self.num_classes: int = num_classes
        
        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=img_size * img_size)
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for discriminating real/fake images based on labels.
        
        Args:
            images (torch.Tensor): Input images, shape (batch_size, 1, img_size, img_size).
            labels (torch.Tensor): Class labels, shape (batch_size,).
            
        Returns:
            torch.Tensor: Probability of images being real.
        """
        embedded_labels = self.label_embedding(labels).view(-1, 1, self.img_size, self.img_size)
        combined_input = torch.cat([images, embedded_labels], dim=1)
        return self.network(combined_input).view(-1)
