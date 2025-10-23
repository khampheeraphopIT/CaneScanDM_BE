import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import requests
import time
from io import BytesIO
from datetime import datetime
import shap
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm
from datetime import datetime
from src.config import settings
from src.utils.logger import logger

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 32
NUM_CLASSES = 6
NUMERICAL_FEATURES = [
    'Temperature', 'Humidity_PER', 'Rainfall',
    'VARI', 'ExG', 'CIVE',
    'GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_Energy',
    'LBP_Feature', 'Edge_Density'
]

# โหลดข้อมูลจังหวัดจาก api_province.json
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "api_province.json"), "r", encoding="utf-8") as f:
    provinces_data = json.load(f)

THAI_PROVINCES_MAPPING = {prov["name_th"].lower(): prov["name_en"] for prov in provinces_data}
THAI_PROVINCES_MAPPING.update({
    "กรุงเทพมหานคร": "Bangkok",
    "ลพบุรี": "Lopburi",
    "หนองบัวลำภู": "Nong Bua Lamphu",
    "พังงา": "Phang Nga",
    "บึงกาฬ": "Bueng Kan",
})

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomResizedCrop(size=128, scale=(0.7, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Focal Loss
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

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, numerical_data, transform=None):
        valid_indices = []
        for idx, path in enumerate(image_paths):
            if os.path.exists(path):
                valid_indices.append(idx)
            else:
                logger.warning(f"Image not found: {path}, skipping this sample")
        self.image_paths = image_paths[valid_indices]
        self.labels = labels[valid_indices]
        self.numerical_data = numerical_data[valid_indices]
        self.transform = transform
        logger.info(f"Dataset size after filtering missing images: {len(self.image_paths)}")

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
            raise ValueError(f"Failed to load image {image_path}")
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'numerical': torch.tensor(numerical, dtype=torch.float32)
        }, torch.tensor(label, dtype=torch.long)

# Custom Model (ซ้ำกับ inference.py แต่เก็บไว้สำหรับ training)
class CustomModel(nn.Module):
    def __init__(self, num_numerical_features=len(NUMERICAL_FEATURES), num_classes=NUM_CLASSES):
        super(CustomModel, self).__init__()
        self.base_model = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.numerical_fc = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(num_ftrs + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, numerical):
        img_features = self.base_model(images)
        num_features = self.numerical_fc(numerical)
        combined = torch.cat((img_features, num_features), dim=1)
        output = self.final_fc(combined)
        return output

# Weather data handling
def get_weather_data(province):
    province_lower = province.lower()
    if province_lower in THAI_PROVINCES_MAPPING:
        province_mapped = THAI_PROVINCES_MAPPING[province_lower]
    else:
        logger.warning(f"Province {province} not found in mapping, using as is")
        province_mapped = province
    api_key = settings.OPENWEATHER_API_KEY
    url = f"http://api.openweathermap.org/data/2.5/weather?q={province_mapped},TH&appid={api_key}&units=metric"
    try:
        time.sleep(0.1)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['cod'] != 200:
            logger.error(f"ไม่พบข้อมูลสภาพอากาศสำหรับ {province_mapped}: {data['message']}")
            return [30.0, 70.0, 0.0], {"temperature": 30.0, "humidity": 70.0, "rainfall": 0.0}
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        rainfall = data.get('rain', {}).get('1h', 0.0)
        logger.info(f"Weather data for {province_mapped}: Temp={temperature}°C, Humidity={humidity}%, Rainfall={rainfall}mm")
        return [temperature, humidity, rainfall], {"temperature": temperature, "humidity": humidity, "rainfall": rainfall}
    except Exception as e:
        logger.error(f"Failed to fetch weather data for {province_mapped}: {e}")
        return [30.0, 70.0, 0.0], {"temperature": 30.0, "humidity": 70.0, "rainfall": 0.0}

# Feature extraction functions
def calculate_vegetation_indices(image_path):
    if not os.path.exists(image_path):
        return {'vari': 0.0, 'exg': 0.0, 'cive': 0.0}
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image for vegetation indices: {image_path}")
        return {'vari': 0.0, 'exg': 0.0, 'cive': 0.0}
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image)
    vari = (g - r) / (g + r - b + 1e-10)
    exg = 2 * g - r - b
    cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
    return {
        'vari': np.mean(vari),
        'exg': np.mean(exg),
        'cive': np.mean(cive)
    }

def calculate_glcm_features(image_path):
    if not os.path.exists(image_path):
        return {'contrast': 0.0, 'homogeneity': 0.0, 'energy': 0.0}
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Failed to load image for GLCM features: {image_path}")
        return {'contrast': 0.0, 'homogeneity': 0.0, 'energy': 0.0}
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy
    }

def calculate_lbp_feature(image_path):
    if not os.path.exists(image_path):
        return 0.0
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Failed to load image for LBP feature: {image_path}")
        return 0.0
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist.mean()

def calculate_edge_density(image_path):
    if not os.path.exists(image_path):
        return 0.0
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Failed to load image for edge density: {image_path}")
        return 0.0
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges) / (image.shape[0] * image.shape[1])

def compute_all_features(image_path, province=None):
    if province:
        weather_features, weather_dict = get_weather_data(province)
    else:
        weather_features = [30.0, 70.0, 0.0]
        weather_dict = {"temperature": 30.0, "humidity": 70.0, "rainfall": 0.0}
    vegetation_indices = calculate_vegetation_indices(image_path)
    glcm_features = calculate_glcm_features(image_path)
    lbp_feature = calculate_lbp_feature(image_path)
    edge_density = calculate_edge_density(image_path)
    features = [
        weather_features[0],
        weather_features[1],
        weather_features[2],
        vegetation_indices['vari'],
        vegetation_indices['exg'],
        vegetation_indices['cive'],
        glcm_features['contrast'],
        glcm_features['homogeneity'],
        glcm_features['energy'],
        lbp_feature,
        edge_density
    ]
    scaler = MinMaxScaler()
    features = scaler.fit_transform(np.array(features).reshape(1, -1)).flatten()
    return features, weather_dict

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    images_dir = os.path.abspath(os.path.join(os.path.dirname(csv_path), "src/images"))
    df['Image_URL'] = df['Image_URL'].apply(lambda x: os.path.join(images_dir, os.path.basename(x)))
    logger.info(f"จำนวนแถวก่อนลบลิงก์ซ้ำ: {len(df)}")
    df = df.drop_duplicates(subset=['Image_URL'])
    logger.info(f"จำนวนแถวหลังลบลิงก์ซ้ำ: {len(df)}")
    df['Disease'] = df['Disease'].str.capitalize()
    label_map = {"Healthy": 0, "Yellow": 1, "Rust": 2, "Redrot": 3, "Mosaic": 4, "Notsugarcane": 5}
    df['label'] = df['Disease'].map(label_map)
    if df['label'].isna().sum() > 0:
        logger.warning(f"พบ {df['label'].isna().sum()} แถวที่ label เป็น NaN. ลบแถวเหล่านั้น.")
        df = df[df['label'].notna()]
        df['label'] = df['label'].astype(int)
        if len(df) == 0:
            raise ValueError("Dataset ว่างหลังจากลบ label ไม่ถูกต้อง. ตรวจสอบคอลัมน์ 'Disease'.")
    logger.info("Class distribution before balancing:")
    logger.info(df['Disease'].value_counts())
    has_province = 'Province' in df.columns
    if not has_province:
        logger.warning("คอลัมน์ 'Province' ไม่พบใน dataset.csv จะใช้ province จาก Frontend แทน")
    numerical_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing numerical features"):
        image_path = row['Image_URL']
        logger.info(f"Processing image: {image_path}")
        province = row.get('Province') if has_province else None
        features, _ = compute_all_features(image_path, province)
        numerical_data.append(features)
    numerical_data = np.array(numerical_data)
    image_paths = df['Image_URL'].values
    labels = df['label'].values.astype(int)
    logger.info("Finished computing numerical features")
    return image_paths, labels, numerical_data

def analyze_feature_importance(model, val_loader, device, feature_names):
    model.eval()
    background_data = []
    for batch in val_loader:
        inputs, _ = batch
        numerical = inputs['numerical'].to(device)
        background_data.append(numerical.cpu().numpy())
        if len(background_data) * BATCH_SIZE >= 100:
            break
    background_data = np.concatenate(background_data, axis=0)
    def model_wrapper(numerical):
        numerical_tensor = torch.tensor(numerical, dtype=torch.float32).to(device)
        images_tensor = torch.zeros((numerical.shape[0], 3, 128, 128)).to(device)
        with torch.no_grad():
            outputs = model(images_tensor, numerical_tensor)
        return outputs.cpu().numpy()
    explainer = shap.KernelExplainer(model_wrapper, background_data)
    shap_values = explainer.shap_values(background_data[:10])
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, background_data[:10], feature_names=feature_names, plot_type="bar")
    plt.title("Feature Importance for Disease Prediction")
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png")
    plt.close()

def train_model(csv_path, model_dir='model', fine_tune=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    X, y, num_data = prepare_data(csv_path)
    if np.any(np.isnan(y)):
        raise ValueError("Labels (y) contain NaN values after prepare_data. Please check the dataset.")
    class_counts = np.bincount(y)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    X_train, X_val, y_train, y_val, num_train, num_val = train_test_split(
        X, y, num_data, test_size=0.2, stratify=y, random_state=42
    )
    train_dataset = CustomDataset(X_train, y_train, num_train, transform=train_transform)
    val_dataset = CustomDataset(X_val, y_val, num_val, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model = CustomModel().to(device)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    num_epochs = 30
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{timestamp}.pth")
    logger.info(f"Model will be saved as: {model_path}")
    with open("training_metrics.csv", "w") as f:
        f.write("Epoch,Train Loss,Train Acc,Train F1,Val Loss,Val Acc,Val F1,Val Precision,Val Recall\n")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch
            images = inputs['image'].to(device)
            numerical = inputs['numerical'].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, numerical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        train_acc = train_correct / train_total
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = batch
                images = inputs['image'].to(device)
                numerical = inputs['numerical'].to(device)
                labels = labels.to(device)
                outputs = model(images, numerical)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                    f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
                    f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
                    f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        with open("training_metrics.csv", "a") as f:
            f.write(f"{epoch+1},{train_loss/len(train_loader)},{train_acc},{train_f1},"
                    f"{val_loss/len(val_loader)},{val_acc},{val_f1},{val_precision},{val_recall}\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model: {model_path} with val_loss: {val_loss/len(val_loader):.4f}, val_acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        scheduler.step(val_loss)
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break
    logger.info("Analyzing feature importance...")
    analyze_feature_importance(model, val_loader, device, NUMERICAL_FEATURES)
    logger.info(f"Training finished. Best validation accuracy: {best_val_acc*100:.2f}%")
    logger.info(f"Final model saved as: {model_path}")
    return model, model_path

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    model = train_model("../../dataset_updated.csv", fine_tune=False)