# Training Module

This module contains the training scripts for various models used in the MedMNIST project.

## Scripts

*   `classifier.py`: Trains the ResNet18 classifier on primary or synthetic data for evaluation.
*   `diffusion.py`: Trains the Conditional Diffusion Model.
*   `gan.py`: Trains the Generative Adversarial Network (GAN).
*   `latent_diffusion.py`: Trains the Latent Diffusion Model (LDM).
*   `uncond.py`: Trains the unconditional diffusion baseline model.
*   `vae.py`: Trains the Variational Autoencoder (VAE) used to extract latent representations.

## Usage

Each script is designed to be executed directly as a standalone program. Some take command-line arguments (like `classifier.py`), while others use hardcoded hyperparameter configurations inside the script.

```bash
# Example
python -m src.training.classifier --name "baseline"
```
