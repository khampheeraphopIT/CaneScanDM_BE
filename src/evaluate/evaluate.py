import os
import torch
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model.model_arch import SugarcaneDiseaseModel
from src.model.model import CustomDataset, val_transform
from src.config import settings

def evaluate_model():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 1. Paths
    CSV_PATH = "dataset_synced.csv"

    MODEL_PATH = os.path.join("model", "best_model.pth")
    SCALER_PATH = os.path.join("model", "scaler.joblib")
    RESULTS_DIR = "evaluation_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 2. Load Data (Keep consistent with model.py)
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    images_dir = os.path.abspath(os.path.join(os.path.dirname(CSV_PATH), "src/images"))
    df['Image_URL'] = df['Image_URL'].apply(lambda x: os.path.join(images_dir, os.path.basename(x)))
    df = df[df['Image_URL'].apply(os.path.exists)]
    df = df.drop_duplicates(subset=['Image_URL']).dropna(subset=['Disease'])

    label_map = {"Healthy": 0, "Yellow": 1, "Rust": 2, "Redrot": 3, "Mosaic": 4, "Notsugarcane": 5}
    df['label'] = df['Disease'].str.capitalize().map(label_map)
    df = df.dropna(subset=['label'])

    feature_cols = ['Temperature', 'Humidity_PER', 'Rainfall',
                    'VARI', 'ExG', 'CIVE', 'GLCM_Contrast', 
                    'GLCM_Homogeneity', 'GLCM_Energy', 
                    'LBP_Feature', 'Edge_Density']
    
    X_img_paths = df['Image_URL'].values
    y_labels = df['label'].values.astype(int)
    numerical_raw = df[feature_cols].values

    # 3. Split & Scale (Must match model.py split logic)
    indices = np.arange(len(y_labels))
    _, val_idx = train_test_split(indices, test_size=0.2, stratify=y_labels, random_state=42)
    
    scaler = joblib.load(SCALER_PATH)
    num_val = scaler.transform(numerical_raw[val_idx])

    val_ds = CustomDataset(X_img_paths[val_idx], y_labels[val_idx], num_val, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 4. Load Model
    print("Loading model...")
    model = SugarcaneDiseaseModel(num_numerical_features=len(feature_cols), num_classes=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 5. Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch, (target) in tqdm(val_loader, desc="Evaluating"):
            imgs = batch['image'].to(DEVICE)
            nums = batch['numerical'].to(DEVICE)
            target = target.to(DEVICE)

            output = model(imgs, nums)
            preds = output.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # 6. Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, 
                                 target_names=list(label_map.keys()), 
                                 zero_division=0)
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(report)

    # 7. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_map.keys()), 
                yticklabels=list(label_map.keys()))
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

if __name__ == "__main__":
    evaluate_model()
