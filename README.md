# Embedding-Guided Diffusion for Medical Image Augmentation (MedMNIST)

This project explores the use of **Conditional Diffusion Models** (Embedding-Guided) to solve data scarcity in medical imaging, specifically using the **PneumoniaMNIST** dataset from MedMNIST.

## 🚀 Overview

Medical image datasets often suffer from extreme imbalance or scarcity. Traditional augmentation (flips, rotations) can distort anatomical structures. Generative models like GANs often lack diversity (mode collapse). This project proposes an **Embedding-Guided Diffusion** approach using **Cross-Attention** to generate high-quality, pathologically accurate synthetic data.

## 📊 Key Results

| Scenario                        | Accuracy   | AUC-ROC    | Result            |
| :------------------------------ | :--------- | :--------- | :---------------- |
| **Proposed (Guided Diffusion)** | **0.8830** | **0.9704** | ✅ Best Trade-off |
| **Baseline (10% Data)**         | 0.8894     | 0.9682     | Overfits          |
| **Traditional Aug**             | 0.8349     | 0.9706     | Distorts anatomy  |
| **GAN Augment**                 | 0.8446     | 0.9666     | Low quality       |
| **Unconditional Diffusion**     | 0.8526     | 0.9526     | Label Noise       |

## 🛠️ Features

- **Conditional Diffusion**: UNet2DConditionModel with Cross-Attention for label guidance.
- **Scarcity Simulation**: Only 10% of training data used to simulate real-world medical constraints.
- **Ablation Studies**: Comprehensive comparison with GANs, Traditional Aug, and Unconditional Diffusion.
- **MedMNIST Integration**: Automated ETL for MedMNIST datasets.

## 📁 Structure

- `src/models/`: Diffusion and GAN architectures.
- `src/train_diffusion.py`: Training script with HF Accelerate.
- `src/train_classifier.py`: Benchmark evaluation script.
- `reports/final_report.md`: Detailed research findings.

## ⚙️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/EOV-ChinhQD/MedMNIST.git
   cd MedMNIST
   ```
2. Create environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Generate synthetic data from Hugging Face:
   [Link to model on HF Hub](https://huggingface.co/Kenkaw303/medmnist-pneumonia-diffusion)

## 📝 Citation

This project is part of a research evaluation on generative augmentation for medical AI.
