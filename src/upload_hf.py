import os
from huggingface_hub import HfApi, create_repo
import argparse

def upload_to_hf(model_path, repo_name, token):
    api = HfApi(token=token)
    
    # Create repo if not exists
    try:
        user = api.whoami()['name']
        repo_id = f"{user}/{repo_name}"
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"Repo {repo_id} is ready.")
    except Exception as e:
        print(f"Error creating/finding repo: {e}")
        return

    # Upload
    print(f"Uploading {model_path} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print("Upload successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/diffusion_v1/final_model.pt")
    parser.add_argument("--repo", type=str, required=True, help="e.g. medmnist-diffusion")
    parser.add_argument("--token", type=str, required=True, help="Your HF Write Token")
    args = parser.parse_args()
    
    upload_to_hf(args.model, args.repo, args.token)
