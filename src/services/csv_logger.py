import os
import pandas as pd
from src.config import settings
from src.utils.logger import logger

def save_upload_to_csv(upload_data):
    data = {
        'Timestamp': [upload_data['timestamp']],
        'Image_Path': [upload_data['image_path']],
        'Disease': [upload_data['prediction']['disease']],
        'Confidence': [upload_data['prediction']['confidence']],
        'Risk_Level': [upload_data['prediction']['risk_level']],
        'Province': [upload_data['province']],
        'Temperature': [upload_data['temperature']],
        'Humidity': [upload_data['humidity']],
        'Rainfall': [upload_data['rainfall']]
    }
    df = pd.DataFrame(data)

    try:
        if os.path.exists(settings.CSV_LOG_PATH):
            df.to_csv(settings.CSV_LOG_PATH, mode='a', header=False, index=False)
        else:
            df.to_csv(settings.CSV_LOG_PATH, mode='w', header=True, index=False)
        logger.info(f"Saved prediction to {settings.CSV_LOG_PATH}")
    except Exception as e:
        logger.error(f"Failed to save to CSV: {e}")
