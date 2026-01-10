import pandas as pd
import numpy as np
import os

def analyze_dataset(csv_path):
    print(f"--- Analyzing Dataset: {csv_path} ---")
    df = pd.read_csv(csv_path)
    
    # 1. Basic Stats
    print(f"Total rows: {len(df)}")
    
    # 2. Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\n[WARNING] Missing values found:")
        print(missing[missing > 0])
    else:
        print("\n[OK] No missing values found.")
        
    # 3. Analyze Feature Distributions
    feature_cols = ['Temperature', 'Humidity_PER', 'Rainfall',
                    'VARI', 'ExG', 'CIVE', 'GLCM_Contrast', 
                    'GLCM_Homogeneity', 'GLCM_Energy', 
                    'LBP_Feature', 'Edge_Density']
    
    print("\n--- Feature Statistics ---")
    stats = df[feature_cols].describe().transpose()[['min', 'max', 'mean', 'std']]
    print(stats)
    
    # 4. Check for extreme outliers (Z-score > 3)
    print("\n--- Potential Outliers (Z-score > 3) ---")
    for col in feature_cols:
        col_data = df[col]
        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            print(f"{col}: {outliers} potential outliers ({(outliers/len(df))*100:.2f}%)")

    # 5. Check Image Paths Existence
    print("\n--- Image File Verification ---")
    images_dir = os.path.abspath(os.path.join(os.path.dirname(csv_path), "src/images"))
    def check_path(url):
        p = os.path.join(images_dir, os.path.basename(url))
        return os.path.exists(p)
    
    exists_count = df['Image_URL'].apply(check_path).sum()
    print(f"Files existing on disk: {exists_count} / {len(df)}")
    if exists_count < len(df):
        print(f"[WARNING] {len(df) - exists_count} images are missing from disk!")

if __name__ == "__main__":
    analyze_dataset('dataset_updated.csv')
