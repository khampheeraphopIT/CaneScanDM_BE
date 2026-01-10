import torch
import joblib
import numpy as np
from PIL import Image
from src.model.inference import load_inference_assets, compute_image_features, val_transform
from src.utils.constants import reverse_label_map

def debug_prediction(image_path):
    model, scaler, device = load_inference_assets()
    
    # 1. Raw Image Processing
    image = Image.open(image_path).convert('RGB')
    img_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # 2. Features
    img_feat = compute_image_features(image_path)
    weather_feat = [30.0, 70.0, 0.0] # Dummy weather
    raw_nums = np.array([weather_feat + img_feat])
    
    # 3. Scaling
    norm_nums = scaler.transform(raw_nums)
    num_tensor = torch.tensor(norm_nums, dtype=torch.float32).to(device)
    
    print(f"Raw Features: {img_feat}")
    print(f"Scaled Numerical: {norm_nums}")
    
    # 4. Predict
    with torch.no_grad():
        output = model(img_tensor, num_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        
    print(f"Output Probs: {probs}")
    print(f"Prediction: {reverse_label_map[pred.item()]} ({conf.item()*100:.2f}%)")

if __name__ == "__main__":
    # Test with user image
    debug_prediction('C:/Users/poplo/.gemini/antigravity/brain/3f86a0dd-5075-4833-b631-5b3dbe3c5ade/uploaded_image_1767601991986.png')
