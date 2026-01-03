import os
import torch
import json
from PIL import Image
import numpy as np
import cv2
import requests
import time
import joblib
from torchvision import transforms
from src.utils.logger import logger
from src.utils.constants import reverse_label_map
from src.config import settings
from src.model.model_arch import SugarcaneDiseaseModel

# Constants
IMAGE_SIZE = 224

# Texture enhancement helper (must match training)
def apply_clahe(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_cv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_cv = cv2.merge((cl, a, b))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_cv)

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: apply_clahe(img)),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extraction for inference
def compute_image_features(image_path):
    if not os.path.exists(image_path): return [0.0] * 8
    image = cv2.imread(image_path)
    if image is None: return [0.0] * 8
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb.astype(float))
    vari = np.mean((g - r) / (g + r - b + 1e-10))
    exg = np.mean(2 * g - r - b)
    cive = np.mean(0.441 * r - 0.811 * g + 0.385 * b + 18.78745)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_feat = (hist.astype("float") / (hist.sum() + 1e-7)).mean()
    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    
    return [vari, exg, cive, contrast, homogeneity, energy, lbp_feat, edge_density]

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

def get_weather_data(province):
    province_lower = province.lower()
    province_mapped = THAI_PROVINCES_MAPPING.get(province_lower, province)

    api_key = settings.OPENWEATHER_API_KEY
    if not api_key:
        logger.warning("No OpenWeather API key found, using defaults.")
        return [30.0, 70.0, 0.0], {"temperature": 30.0, "humidity": 70.0, "rainfall": 0.0}

    url = f"http://api.openweathermap.org/data/2.5/weather?q={province_mapped},TH&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        rainfall = data.get('rain', {}).get('1h', 0.0)
        return [temperature, humidity, rainfall], {"temperature": temperature, "humidity": humidity, "rainfall": rainfall}
    except Exception as e:
        logger.error(f"Failed to fetch weather for {province_mapped}: {e}")
        return [30.0, 70.0, 0.0], {"temperature": 30.0, "humidity": 70.0, "rainfall": 0.0}

def predict_single_image(model, scaler, device, image_path, province):
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = val_transform(image).unsqueeze(0).to(device)
        
        weather_feat, weather_dict = get_weather_data(province)
        img_feat = compute_image_features(image_path)
        raw_nums = np.array([weather_feat + img_feat])
        
        if scaler:
            norm_nums = scaler.transform(raw_nums)
        else:
            norm_nums = raw_nums
            
        num_tensor = torch.tensor(norm_nums, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(img_tensor, num_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        
        disease = reverse_label_map[pred.item()]
        return {
            "disease": disease,
            "confidence": conf.item(),
            "probabilities": {reverse_label_map[i]: f"{p*100:.2f}%" for i, p in enumerate(probs[0].cpu().numpy())},
            "weather": weather_dict
        }
    except Exception as e:
        logger.error(f"Prediction error for {image_path}: {e}")
        return {"error": str(e)}

def predict_image_batch(image_paths: list, provinces: list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SugarcaneDiseaseModel().to(device)
    
    # Load best model and scaler from settings
    if os.path.exists(settings.MODEL_PATH):
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
    model.eval()
    
    scaler = None
    if os.path.exists(settings.SCALER_PATH):
        scaler = joblib.load(settings.SCALER_PATH)

    results = []
    for img_p, prov in zip(image_paths, provinces):
        res = predict_single_image(model, scaler, device, img_p, prov)
        results.append(res)
            
    return results
