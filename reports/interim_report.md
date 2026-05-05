# Interim Progress Report: Embedding-Guided Diffusion on MedMNIST

## 1. Accomplishments

- [x] **Environment Setup**: Python venv created, all dependencies (`diffusers`, `accelerate`, `medmnist`, `torch`) installed.
- [x] **Data ETL**: Downloaded `PneumoniaMNIST` at 128x128. Created a 10% "scarcity" split (470 samples) stratified by class.
- [x] **EDA**: Visualized samples and computed normalization stats (Mean=0.5742, Std=0.1773).
- [x] **Baselines**:
  - **Baseline (10% data)**: ACC: 0.8894, AUC: 0.9682.
  - **Traditional Aug**: ACC: 0.8349, AUC: 0.9706.
- [x] **Proposed Method Implementation**:
  - [x] UNet-based Conditional Diffusion Model (Cross-Attention on Label Embeddings).
  - [x] Training script with `accelerate` and `diffusers`.

## 2. Current Status

- **Diffusion Training**: In progress (currently at Epoch 20/200).
- **GPU Usage**: RTX 3060 (10.2 GB / 12 GB used).

## 3. Preliminary Benchmarks

| Scenario | ACC    | F1     | AUC    |
| :------- | :----- | :----- | :----- |
| trad_aug | 0.8349 | 0.8831 | 0.9706 |
| baseline | 0.8894 | 0.918  | 0.9682 |

## 4. Next Steps

1. Complete Diffusion training (at least 100 epochs for decent quality).
2. Generate synthetic images (Normal vs Pneumonia).
3. Evaluate Generative Quality (FID).
4. Train Classifier on **Baseline + Synthetic** data.
5. Final comparison and report.
