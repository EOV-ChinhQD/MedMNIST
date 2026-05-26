import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import os
from src.models.gan import Generator, Discriminator
from src.data.dataset import MedMNISTProcessed
from src.utils.logger import setup_logger
from tqdm.auto import tqdm

def train_gan() -> None:
    """
    Trains the Generative Adversarial Network (GAN).
    """
    # Config
    lr: float = 0.0002
    beta1: float = 0.5
    num_epochs: int = 100
    batch_size: int = 32
    latent_dim: int = 100
    img_size: int = 128
    output_dir: str = "artifacts/gan_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger('gan_logger', 'logs/train_gan.log')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    netG = Generator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)
    
    # Loss & Optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Data
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MedMNISTProcessed('data/processed/train_10.npz', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    real_label: float = 1.0
    fake_label: float = 0.0

    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(dataloader):
            bs: int = data.size(0)
            data = data.to(device)
            labels = labels.to(device)

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            output = netD(data, labels)
            errD_real = criterion(output, torch.full((bs,), real_label, device=device))
            errD_real.backward()

            noise = torch.randn(bs, latent_dim, 1, 1, device=device)
            fake = netG(noise, labels)
            output = netD(fake.detach(), labels)
            errD_fake = criterion(output, torch.full((bs,), fake_label, device=device))
            errD_fake.backward()
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            output = netD(fake, labels)
            errG = criterion(output, torch.full((bs,), real_label, device=device))
            errG.backward()
            optimizerG.step()

        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch} Loss_D: {errD_real.item()+errD_fake.item():.4f} Loss_G: {errG.item():.4f}")
            # Save samples
            utils.save_image(fake.detach()[:16], f"{output_dir}/samples_epoch_{epoch}.png", normalize=True)

    torch.save(netG.state_dict(), f"{output_dir}/final_netG.pt")
    logger.info("GAN training finished.")

if __name__ == "__main__":
    train_gan()
