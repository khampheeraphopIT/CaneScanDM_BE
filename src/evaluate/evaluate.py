import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- เพิ่ม path ของ parent folder เพื่อ import model ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- import โมเดลและ dataset ของคุณ ---
from model.model import CustomModel, CustomDataset, val_transform, prepare_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Paths ---
CSV_PATH = "../../dataset_updated.csv"
MODEL_PATH = "../model/model/model_20251019_122736.pth"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- เตรียมข้อมูล ---
print("Preparing data...")
image_paths, labels, numerical_data = prepare_data(CSV_PATH)

from sklearn.model_selection import train_test_split
_, X_val, _, y_val, _, num_val = train_test_split(
    image_paths, labels, numerical_data, test_size=0.2, stratify=labels, random_state=42
)

val_dataset = CustomDataset(X_val, y_val, num_val, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- โหลด model ---
print("Loading model...")
model = CustomModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Evaluate ---
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        inputs, labels_batch = batch
        images = inputs['image'].to(DEVICE)
        numerical = inputs['numerical'].to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        outputs = model(images, numerical)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

# --- Metrics ---
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# --- Classification report ---
class_names = ['Healthy','Yellow','Rust','RedRot','Mosaic','NotSugarcane']
report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
print("\nClassification Report:\n")
print(report)

# --- Save results CSV ---
results_df = pd.DataFrame({
    'Image': [val_dataset.image_paths[i] for i in range(len(val_dataset))],
    'True_Label': [val_dataset.labels[i] for i in range(len(val_dataset))],
    'Pred_Label': all_preds
})
results_csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Saved evaluation results to {results_csv_path}")

# --- Confusion Matrix Heatmap ---
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
heatmap_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(heatmap_path)
plt.close()
print(f"Saved confusion matrix heatmap to {heatmap_path}")
