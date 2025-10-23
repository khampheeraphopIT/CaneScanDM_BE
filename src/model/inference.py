import os
import torch
import json
from PIL import Image
from fastapi import HTTPException
from src.utils.logger import logger
from src.utils.constants import reverse_label_map
from src.config import settings
from torchvision import models, transforms
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
import requests
import time
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.preprocessing import MinMaxScaler

# Constants
NUMERICAL_FEATURES = [
    'Temperature', 'Humidity_PER', 'Rainfall',
    'VARI', 'ExG', 'CIVE',
    'GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_Energy',
    'LBP_Feature', 'Edge_Density'
]

# Data transforms
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Model
class CustomModel(torch.nn.Module):
    def __init__(self, num_numerical_features=len(NUMERICAL_FEATURES), num_classes=6):
        super(CustomModel, self).__init__()
        self.base_model = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = torch.nn.Identity()
        self.numerical_fc = torch.nn.Sequential(
            torch.nn.Linear(num_numerical_features, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.final_fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs + 32, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, images, numerical):
        img_features = self.base_model(images)
        num_features = self.numerical_fc(numerical)
        combined = torch.cat((img_features, num_features), dim=1)
        output = self.final_fc(combined)
        return output

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

def predict_image(image_path: str, province: str):
    device = torch.device('cpu')  # Force CPU for production
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = val_transform(image).unsqueeze(0).to(device)
        numerical_data, weather_data = compute_all_features(image_path, province)
        numerical = torch.tensor(numerical_data, dtype=torch.float32).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preparing data: {e}")

    model = CustomModel()
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image_tensor, numerical)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    disease = reverse_label_map[predicted.item()]
    confidence_value = confidence.item()
    probabilities_dict = {
        reverse_label_map[i]: f"{prob * 100:.2f}%"
        for i, prob in enumerate(probabilities[0].cpu().numpy())
    }

    return disease, confidence_value, probabilities_dict, weather_data

def visualize_gradcam(model, image_path, device, output_path="gradcam_output.jpg"):
    model.eval()
    target_layer = model.base_model.layer4[-1]
    image = Image.open(image_path).convert('RGB')
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    cam = GradCAM(model=model.base_model, target_layers=[target_layer], use_cuda=(device.type == 'cuda'))
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    img = np.array(image.resize((128, 128))) / 255.0
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    logger.info(f"Grad-CAM visualization saved as {output_path}")
    return output_path