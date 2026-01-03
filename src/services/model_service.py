import torch
from src.utils.logger import logger
from src.model.inference import predict_image_batch

def predict_service(image_paths: list, provinces: list):
    try:
        results = predict_image_batch(image_paths, provinces)
        logger.info(f"Processed batch of {len(image_paths)} images")
        return results
    except Exception as e:
        logger.error(f"Failed prediction batch: {e}")
        return None
