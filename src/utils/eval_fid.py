import os
import numpy as np
from cleanfid import fid
from PIL import Image
import shutil

def calculate_fid(real_npz, synth_npz, tmp_dir='tmp_fid'):
    # clean-fid expects directories of images
    real_dir = os.path.join(tmp_dir, 'real')
    synth_dir = os.path.join(tmp_dir, 'synth')
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)
    
    # Save real images
    real_data = np.load(real_npz)['images']
    for i, img in enumerate(real_data):
        Image.fromarray(img).save(os.path.join(real_dir, f"{i}.png"))
        
    # Save synth images
    synth_data = np.load(synth_npz)['images']
    for i, img in enumerate(synth_data):
        Image.fromarray(img).save(os.path.join(synth_dir, f"{i}.png"))
        
    # Calculate FID
    score = fid.compute_fid(real_dir, synth_dir)
    print(f"FID Score: {score}")
    
    # Cleanup
    shutil.rmtree(tmp_dir)
    return score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--synth", type=str, required=True)
    args = parser.parse_args()
    
    calculate_fid(args.real, args.synth)
