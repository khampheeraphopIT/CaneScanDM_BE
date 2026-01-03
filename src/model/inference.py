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
    transforms.Lambda(apply_clahe),
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

# Mapping จากข้อมูลผู้ใช้เพื่อแก้ปัญหาการสะกดไม่ตรงกัน
ENGLISH_TO_THAI = {
    "Bangkok": "กรุงเทพมหานคร", "Krung Thep Maha Nakhon": "กรุงเทพมหานคร",
    "Krabi": "กระบี่", "Kanchanaburi": "กาญจนบุรี", "Kalasin": "กาฬสินธุ์",
    "Kamphaeng Phet": "กำแพงเพชร", "Khon Kaen": "ขอนแก่น", "Chanthaburi": "จันทบุรี",
    "Chachoengsao": "ฉะเชิงเทรา", "Chonburi": "ชลบุรี", "Chon Buri": "ชลบุรี",
    "Chai Nat": "ชัยนาท", "Chaiyaphum": "ชัยภูมิ", "Chumphon": "ชุมพร",
    "Trang": "ตรัง", "Trat": "ตราด", "Tak": "ตาก", "Nakhon Nayok": "นครนายก",
    "Nakhon Pathom": "นครปฐม", "Nakhon Phanom": "นครพนม", "Nakhon Ratchasima": "นครราชสีมา",
    "Nakhon Si Thammarat": "นครศรีธรรมราช", "Nakhon Sawan": "นครสวรรค์",
    "Nonthaburi": "นนทบุรี", "Narathiwat": "นราธิวาส", "Nan": "น่าน",
    "Bueng Kan": "บึงกาฬ", "buogkan": "บึงกาฬ", "Buri Ram": "บุรีรัมย์",
    "Buriram": "บุรีรัมย์", "Pathum Thani": "ปทุมธานี", "Prachuap Khiri Khan": "ประจวบคีรีขันธ์",
    "Prachinburi": "ปราจีนบุรี", "Prachin Buri": "ปราจีนบุรี", "Pattani": "ปัตตานี",
    "Phra Nakhon Si Ayutthaya": "พระนครศรีอยุธยา", "Phayao": "พะเยา", "Phangnga": "พังงา",
    "Phatthalung": "พัทลุง", "Phichit": "พิจิตร", "Phitsanulok": "พิษณุโลก", "Phuket": "ภูเก็ต",
    "Maha Sarakham": "มหาสารคาม", "Mukdahan": "มุกดาหาร", "Yala": "ยะลา", "Yasothon": "ยโสธร",
    "Ranong": "ระนอง", "Rayong": "ระยอง", "Ratchaburi": "ราชบุรี", "Roi Et": "ร้อยเอ็ด",
    "Lopburi": "ลพบุรี", "Loburi": "ลพบุรี", "Lampang": "ลำปาง", "Lamphun": "ลำพูน",
    "Sisaket": "ศรีสะเกษ", "Si Sa Ket": "ศรีสะเกษ", "Sakon Nakhon": "สกลนคร", "Songkhla": "สงขลา",
    "Satun": "สตูล", "Samut Prakan": "สมุทรปราการ", "Samut Songkhram": "สมุทรสงคราม",
    "Samut Sakhon": "สมุทรสาคร", "Saraburi": "สระบุรี", "Sa Kaeo": "สระแก้ว", "Sing Buri": "สิงห์บุรี",
    "Suphan Buri": "สุพรรณบุรี", "Surat Thani": "สุราษฎร์ธานี", "Surin": "สุรินทร์",
    "Sukhothai": "สุโขทัย", "Nong Khai": "หนองคาย", "Nong Bua Lamphu": "หนองบัวลำภู",
    "Amnat Charoen": "อำนาจเจริญ", "Udon Thani": "อุดรธานี", "Uttaradit": "อุตรดิตถ์",
    "Uthai Thani": "อุทัยธานี", "Ubon Ratchathani": "อุบลราชธานี", "Ang Thong": "อ่างทอง",
    "Chiang Rai": "เชียงราย", "Chiang Mai": "เชียงใหม่", "Phetchaburi": "เพชรบุรี",
    "Phetchabun": "เพชรบูรณ์", "Loei": "เลย", "Phrae": "แพร่", "Mae Hong Son": "แม่ฮ่องสอน"
}

# Reverse mapping for API calls (Thai to English)
THAI_TO_ENGLISH = {v: k for k, v in ENGLISH_TO_THAI.items()}

def get_weather_data(province, temperature=None, humidity=None, rainfall=None):
    # ถ้าส่งค่ามาจาก Frontend ให้ใช้ค่านั้นเลย
    if temperature is not None and humidity is not None and rainfall is not None:
        return [float(temperature), float(humidity), float(rainfall)], {
            "temperature": float(temperature),
            "humidity": float(humidity),
            "rainfall": float(rainfall),
            "source": "frontend"
        }

    province_mapped = THAI_TO_ENGLISH.get(province, province)

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

def predict_single_image(model, scaler, device, image_path, province, weather_override=None):
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = val_transform(image).unsqueeze(0).to(device)
        
        if weather_override:
            weather_feat, weather_dict = get_weather_data(
                province, 
                temperature=weather_override.get('temperature'),
                humidity=weather_override.get('humidity'),
                rainfall=weather_override.get('rainfall')
            )
        else:
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

def predict_image_batch(image_paths: list, provinces: list, weather_overrides: list = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SugarcaneDiseaseModel().to(device)
    
    if os.path.exists(settings.MODEL_PATH):
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
    model.eval()
    
    scaler = None
    if os.path.exists(settings.SCALER_PATH):
        scaler = joblib.load(settings.SCALER_PATH)

    results = []
    for i, (img_p, prov) in enumerate(zip(image_paths, provinces)):
        w_override = weather_overrides[i] if weather_overrides else None
        res = predict_single_image(model, scaler, device, img_p, prov, weather_override=w_override)
        results.append(res)
            
    return results
