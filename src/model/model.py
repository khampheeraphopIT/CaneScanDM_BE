import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import requests
import time
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.config import settings
from src.model.model_arch import SugarcaneDiseaseModel, NUMERICAL_FEATURES_COUNT

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 16 # Reduced for ResNet50
IMAGE_SIZE = 224
NUM_CLASSES = 6
NUMERICAL_FEATURES = [
    'Temperature', 'Humidity_PER', 'Rainfall',
    'VARI', 'ExG', 'CIVE',
    'GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_Energy',
    'LBP_Feature', 'Edge_Density'
]

# Texture enhancement helper
def apply_clahe(image):
    # Convert PIL to CV2
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_cv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_cv = cv2.merge((cl, a, b))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_cv)

# Data transforms
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: apply_clahe(img)),
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.2, 1.0)), # Robustness for far/near
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: apply_clahe(img)),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            F_loss = alpha_t * F_loss
        return F_loss.mean()

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, numerical_data, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.numerical_data = numerical_data
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        numerical = self.numerical_data[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a dummy image or handle appropriately. For training, we assume files exist.
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
            
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'numerical': torch.tensor(numerical, dtype=torch.float32)
        }, torch.tensor(label, dtype=torch.long)

# Feature extraction functions (keep these for preparation phase)
def compute_image_features(image_path):
    if not os.path.exists(image_path):
        return [0.0] * 8
    image = cv2.imread(image_path)
    if image is None: return [0.0] * 8
    
    # Vegetation Indices
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb.astype(float))
    vari = np.mean((g - r) / (g + r - b + 1e-10))
    exg = np.mean(2 * g - r - b)
    cive = np.mean(0.441 * r - 0.811 * g + 0.385 * b + 18.78745)
    
    # GLCM
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    # LBP
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_feat = (hist.astype("float") / (hist.sum() + 1e-7)).mean()
    
    # Edge
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    
    return [vari, exg, cive, contrast, homogeneity, energy, lbp_feat, edge_density]

def prepare_and_train(csv_path, model_dir='model'):
    df = pd.read_csv(csv_path)
    images_dir = os.path.abspath(os.path.join(os.path.dirname(csv_path), "src/images"))
    df['Image_URL'] = df['Image_URL'].apply(lambda x: os.path.join(images_dir, os.path.basename(x)))
    df = df.drop_duplicates(subset=['Image_URL']).dropna(subset=['Disease'])
    
    label_map = {"Healthy": 0, "Yellow": 1, "Rust": 2, "Redrot": 3, "Mosaic": 4, "Notsugarcane": 5}
    df['label'] = df['Disease'].str.capitalize().map(label_map)
    df = df.dropna(subset=['label'])
    
    logger.info(f"Dataset summary: {df['Disease'].value_counts().to_dict()}")
    
    # Extract numerical features for all
    X_img_paths = df['Image_URL'].values
    y_labels = df['label'].values.astype(int)
    
    # We simulate weather features if not present in CSV for training consistency
    # In a real scenario, these should be in the dataset_updated.csv
    numerical_raw = []
    for path in tqdm(X_img_paths, desc="Extracting features"):
        # Dummy weather + calculated image features
        weather = [28.0, 75.0, 0.0] 
        img_feats = compute_image_features(path)
        numerical_raw.append(weather + img_feats)
        
    numerical_raw = np.array(numerical_raw)
    
    # Train/Val Split
    indices = np.arange(len(y_labels))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y_labels, random_state=42)
    
    # CRITICAL FIX: Fit Scaler only on Train, then transform both
    scaler = MinMaxScaler()
    num_train = scaler.fit_transform(numerical_raw[train_idx])
    num_val = scaler.transform(numerical_raw[val_idx])
    
    # Save Scaler
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Dataloaders
    train_ds = CustomDataset(X_img_paths[train_idx], y_labels[train_idx], num_train, transform=train_transform)
    val_ds = CustomDataset(X_img_paths[val_idx], y_labels[val_idx], num_val, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SugarcaneDiseaseModel().to(device)
    
    # Class weights for Focal Loss
    counts = np.bincount(y_labels[train_idx])
    weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    weights = weights / weights.sum() * len(counts)
    
    criterion = FocalLoss(gamma=2.0, alpha=weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_val_loss = float('inf')
    num_epochs = 30
    
    for epoch in range(num_epochs):
        model.train()
        t_loss, t_acc = 0, 0
        for batch, (target) in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, nums, target = batch['image'].to(device), batch['numerical'].to(device), target.to(device)
            optimizer.zero_grad()
            output = model(imgs, nums)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_acc += (output.argmax(1) == target).float().mean().item()
            
        # Validation
        model.eval()
        v_loss, v_acc = 0, 0
        with torch.no_grad():
            for batch, (target) in val_loader:
                imgs, nums, target = batch['image'].to(device), batch['numerical'].to(device), target.to(device)
                output = model(imgs, nums)
                v_loss += criterion(output, target).item()
                v_acc += (output.argmax(1) == target).float().mean().item()
        
        v_loss /= len(val_loader)
        v_acc /= len(val_loader)
        logger.info(f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            logger.info("Saved best model.")
            
        scheduler.step(v_loss)

if __name__ == "__main__":
    prepare_and_train("dataset_updated.csv")
