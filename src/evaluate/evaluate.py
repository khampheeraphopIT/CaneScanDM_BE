import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.model import CustomModel, CustomDataset, val_transform  # เปลี่ยนเป็น absolute import
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants (ต้องเหมือนกับใน train.py)
NUMERICAL_FEATURES = [
    'Temperature', 'Humidity_PER', 'Rainfall',
    'VARI', 'ExG', 'CIVE',
    'GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_Energy',
    'LBP_Feature', 'Edge_Density'
]

# Data preparation (เหมือนใน train.py)
def prepare_data(csv_path):
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # แปลงค่าใน Disease ให้เป็นตัวพิมพ์ใหญ่ทั้งหมด
    df['Disease'] = df['Disease'].str.capitalize()
    label_map = {"Healthy": 0, "Yellow": 1, "Rust": 2, "Redrot": 3, "Mosaic": 4, "Notsugarcane": 5}
    df['label'] = df['Disease'].map(label_map)

    # ตรวจสอบและจัดการ NaN ใน label
    if df['label'].isna().any():
        num_na = df['label'].isna().sum()
        df = df.dropna(subset=['label'])
        logger.warning(f"Removed {num_na} rows with NaN labels.")

    image_paths = df['Image_URL'].values
    labels = df['label'].values.astype(int)
    
    # เตรียม numerical data
    numerical_data = np.zeros((len(df), len(NUMERICAL_FEATURES)))
    available_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
    if available_features:
        numerical_data[:, :len(available_features)] = df[available_features].values
    scaler = MinMaxScaler()
    numerical_data = scaler.fit_transform(numerical_data)
    
    logger.info(f"Dataset prepared with {len(image_paths)} samples")
    return image_paths, labels, numerical_data

# Evaluation function
def evaluate_model(csv_path, model_path='model/best_val_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for evaluation: {device}")

    # Prepare data
    image_paths, labels, numerical_data = prepare_data(csv_path)
    _, X_val, _, y_val, _, num_val = train_test_split(
        image_paths, labels, numerical_data, test_size=0.2, stratify=labels, random_state=42
    )

    # Create dataset and dataloader for evaluation
    eval_dataset = CustomDataset(X_val, y_val, num_val, transform=val_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Load the trained model
    model = CustomModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluation loop
    logger.info("Starting evaluation...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(eval_loader, desc="Evaluating"):
            images = inputs['image'].to(device)
            numerical = inputs['numerical'].to(device)
            labels = labels.to(device)
            outputs = model(images, numerical)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    logger.info("Evaluation completed.")

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-Score (weighted): {f1:.4f}")
    logger.info(f"Precision (weighted): {precision:.4f}")
    logger.info(f"Recall (weighted): {recall:.4f}")

    # Classification report
    class_names = ["Healthy", "Yellow", "Rust", "Redrot", "Mosaic", "Notsugarcane"]
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    logger.info("Classification Report:\n" + report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    logger.info("Confusion matrix saved as 'confusion_matrix.png'")

    # Save predictions and true labels to CSV
    results_df = pd.DataFrame({
        'True_Label': all_labels,
        'Predicted_Label': all_preds
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    logger.info("Predictions and true labels saved to 'evaluation_results.csv'")

    return all_preds, all_labels, {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

# Main execution
if __name__ == "__main__":
    csv_path = "dataset_updated.csv"  # ใช้ไฟล์เดียวกับที่เทรน
    model_path = "model/best_val_model.pth"
    predictions, true_labels, metrics = evaluate_model(csv_path, model_path)