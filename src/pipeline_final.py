import os
import subprocess
import argparse

def run_step(cmd, description):
    print(f"\n{'='*20} {description} {'='*20}")
    result = subprocess.run(f"PYTHONPATH=. {cmd}", shell=True)
    if result.returncode != 0:
        print(f"Error in step: {description}")
        exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run generation/eval")
    args = parser.parse_args()

    if not args.skip_train:
        # 1. Huấn luyện VAE với Perceptual Loss (Đảm bảo độ sắc nét)
        run_step("python src/train_vae.py", "Phase A: Training VAE with Perceptual Loss")
        
        # 2. Trích xuất dữ liệu Latent (Nén không gian ảnh)
        run_step("python src/extract_latents.py", "Phase B: Extracting Latent Space Data")
        
        # 3. Huấn luyện Latent Diffusion với CFG
        run_step("python src/train_latent_diffusion.py", "Phase C: Training Latent Diffusion Model (LDM)")

    # 4. Sinh dữ liệu tổng hợp chất lượng cao (CFG 5.0)
    run_step("python src/generate_latent_data.py --n 1000 --output data/synthetic/final_proposed_data.npz", 
             "Phase D: Generating High-Quality Synthetic Data (CFG)")

    # 5. Đánh giá cuối cùng bằng ResNet-18
    run_step("python src/train_classifier.py --name Final_Proposed_LDM --synthetic_path data/synthetic/final_proposed_data.npz", 
             "Step E: Final Downstream Benchmark (ResNet-18)")

    print("\n[SUCCESS] Pipeline completed. Check README.md for the updated metrics.")

if __name__ == "__main__":
    main()
