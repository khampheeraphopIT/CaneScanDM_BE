import torch
from PIL import Image
from fastapi import HTTPException
from src.utils.logger import logger
from src.utils.constants import reverse_label_map
from src.config import settings
from src.model.model import CustomModel, compute_all_features, val_transform, visualize_gradcam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = CustomModel()
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load model")


def predict_image(image_path: str, province: str):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = val_transform(image).unsqueeze(0).to(device)
        numerical_data, weather_data = compute_all_features(image_path, province)
        numerical = torch.tensor(numerical_data, dtype=torch.float32).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preparing data: {e}")

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


def generate_gradcam(image_path: str, output_path: str):
    try:
        visualize_gradcam(model, image_path, device, output_path=output_path)
        logger.info(f"Grad-CAM saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate Grad-CAM: {e}")
        return None
