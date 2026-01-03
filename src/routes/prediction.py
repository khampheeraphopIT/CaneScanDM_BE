from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List
from datetime import datetime
import os

from src.utils.logger import logger
from src.config import settings
from src.services.model_service import predict_service
from src.services.risk_analysis import analyze_risk
from src.services.csv_logger import save_upload_to_csv
from src.routes.province import provinces

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("")
async def predict_disease(
    files: List[UploadFile] = File(...), 
    province: str = Form(...),
    temperature: float = Form(None),
    humidity: float = Form(None),
    rainfall: float = Form(None)
):
    if province not in provinces:
        # Fallback check for Thai name mapping if not in provinces list
        # But for now keep it strict or use the province mapping
        pass

    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    timestamp = datetime.utcnow()
    
    image_paths = []
    for file in files:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
            
        filename, ext = os.path.splitext(file.filename)
        image_filename = f"{filename}_{timestamp.strftime('%H%M%S')}{ext}"
        image_path = os.path.join(settings.UPLOAD_FOLDER, image_filename)
        
        with open(image_path, "wb") as f:
            f.write(await file.read())
        image_paths.append(image_path)

    if not image_paths:
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์ภาพที่ถูกต้อง")

    # Combine weather data if provided
    weather_override = None
    if temperature is not None and humidity is not None and rainfall is not None:
        weather_override = {
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall
        }

    # Call service for batch processing
    predictions = predict_service(
        image_paths, 
        [province] * len(image_paths),
        weather_overrides=[weather_override] * len(image_paths) if weather_override else None
    )
    
    if not predictions:
        raise HTTPException(status_code=500, detail="การวิเคราะห์ล้มเหลว")

    final_results = []
    for i, res in enumerate(predictions):
        if "error" in res:
            final_results.append({"image": os.path.basename(image_paths[i]), "error": res["error"]})
            continue

        disease = res["disease"]
        confidence = res["confidence"]
        weather = res["weather"]
        probabilities = res["probabilities"]
        
        risk_level = analyze_risk(disease, weather)
        
        # Log to CSV
        log_data = {
            "timestamp": timestamp,
            "image_path": image_paths[i],
            "prediction": {"disease": disease, "confidence": confidence, "risk_level": risk_level},
            "province": province,
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "rainfall": weather["rainfall"]
        }
        save_upload_to_csv(log_data)
        
        final_results.append({
            "image": os.path.basename(image_paths[i]),
            "disease": disease,
            "confidence": f"{confidence * 100:.2f}%",
            "risk_level": risk_level,
            "weather": weather,
            "probabilities": probabilities
        })

    return {
        "timestamp": timestamp.isoformat(),
        "province": province,
        "results": final_results
    }
