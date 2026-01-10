import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.model.features import extract_features

def process_one(args):
    img_name, images_dir = args
    full_path = os.path.join(images_dir, img_name)
    if os.path.exists(full_path):
        return extract_features(full_path)
    return [0.0] * 8

def sync_dataset(csv_path):
    print(f"--- Fast Syncing Dataset (Multiprocessing): {csv_path} ---")
    df = pd.read_csv(csv_path)
    images_dir = os.path.abspath(os.path.join(os.path.dirname(csv_path), "src/images"))
    
    feature_cols = ['VARI', 'ExG', 'CIVE', 'GLCM_Contrast', 
                    'GLCM_Homogeneity', 'GLCM_Energy', 
                    'LBP_Feature', 'Edge_Density']
    
    tasks = [(os.path.basename(row['Image_URL']), images_dir) for _, row in df.iterrows()]
    
    recalculated_data = []
    # Using 4 or more workers depends on CPU cores, 4 is usually safe and fast
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        recalculated_data = list(tqdm(executor.map(process_one, tasks), total=len(tasks), desc="Syncing Features"))
            
    # Update DataFrame
    recalculated_data = np.array(recalculated_data)
    for i, col in enumerate(feature_cols):
        df[col] = recalculated_data[:, i]
        
    # Save back
    output_path = "dataset_synced.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[DONE] Synced dataset saved to {output_path}")

if __name__ == "__main__":
    sync_dataset('dataset_updated.csv')
