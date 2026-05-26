# MedMNIST Models Architecture

This directory contains various deep learning architectures tailored for the MedMNIST dataset (primarily targeting 128x128 grayscale images).

## Available Models

- **`classifier.py`**: Contains a modified ResNet18 model for image classification (e.g., distinguishing between normal and pneumonia cases).
- **`diffusion.py`**: Implements a Conditional Diffusion Model (`ConditionalDiffusionModel`) operating directly on pixel space, using class labels for conditional image generation.
- **`gan.py`**: Defines a Conditional Generative Adversarial Network with `Generator` and `Discriminator` classes for synthesizing medical images based on labels.
- **`latent_unet.py`**: Contains `LatentConditionalUNet`, a diffusion model designed to operate on a compressed latent space (16x16) rather than full pixel space.
- **`vae.py`**: Implements a Variational Autoencoder (`MedMNISTVAE`) based on `AutoencoderKL`. It compresses 128x128 images into 16x16 latent representations, which can be utilized by `LatentConditionalUNet`.
- **`medmnist_diffusion.py`**: An inference wrapper (`MedMNISTDiffusion`) that combines the VAE and the Latent UNet. It allows for end-to-end generation from a target class label to a final pixel-space image utilizing classifier-free guidance.
- **`uncond.py`**: Provides an Unconditional Diffusion Model based on `UNet2DModel` for generating images without class conditioning.

## Usage

Most models can be imported and initialized directly. Note that diffusion models (`diffusion.py`, `latent_unet.py`, `uncond.py`) also have corresponding scheduler instantiation functions (`get_scheduler`, `get_latent_scheduler`).
