from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os

from src.utils.logger import logger
from src.config import settings
from src.services.model_service import predict_image, generate_gradcam
from src.services.risk_analysis import analyze_risk
from src.services.csv_logger import save_upload_to_csv
from src.routes.province import provinces

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("")
async def predict_disease(file: UploadFile = File(...), province: str = Form(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์ภาพ (.png, .jpg, .jpeg) เท่านั้น")
    if province not in provinces:
        raise HTTPException(status_code=400, detail="จังหวัดไม่ถูกต้อง")

    timestamp = datetime.utcnow()
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(settings.GRAD_CAM_FOLDER, exist_ok=True)

    filename, ext = os.path.splitext(file.filename)
    image_filename = f"{filename}_{timestamp.strftime('%Y%m%d_%H%M%S')}{ext}"
    image_path = os.path.join(settings.UPLOAD_FOLDER, image_filename)

    with open(image_path, "wb") as f:
        f.write(await file.read())

    disease, confidence, probabilities, weather = predict_image(image_path, province)

    if disease == "Notsugarcane":
        return JSONResponse({
            "error": "ภาพนี้ไม่ใช่ใบอ้อย กรุณาอัปโหลดภาพใบอ้อย",
            "probabilities": probabilities
        }, status_code=400)

    risk_level = analyze_risk(disease, weather)
    gradcam_filename = f"gradcam_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    gradcam_path = os.path.join(settings.GRAD_CAM_FOLDER, gradcam_filename)
    gradcam_result = generate_gradcam(image_path, gradcam_path)

    upload_data = {
        "timestamp": timestamp,
        "image_path": image_path,
        "prediction": {
            "disease": disease,
            "confidence": confidence,
            "risk_level": risk_level
        },
        "province": province,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": weather["rainfall"]
    }
    save_upload_to_csv(upload_data)

    return {
        "timestamp": timestamp.isoformat(),
        "disease": disease,
        "confidence": f"{confidence * 100:.2f}%",
        "risk_level": risk_level,
        "province": province,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": weather["rainfall"],
        "probabilities": probabilities,
        "gradcam_path": gradcam_result or "Grad-CAM generation failed"
    }
