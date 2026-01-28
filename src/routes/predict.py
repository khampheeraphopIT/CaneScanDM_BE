"""
Prediction Route - API endpoint สำหรับวินิจฉัยโรคใบอ้อย
"""
from fastapi import APIRouter, UploadFile, File
from src.services.vision_service import analyze_leaf_image

router = APIRouter()


@router.post("/predict")
async def predict_disease(image: UploadFile = File(...)):
    """
    วินิจฉัยโรคใบอ้อยจากภาพ
    
    - **image**: ไฟล์รูปภาพใบอ้อย (jpg, png, webp)
    
    Returns:
        ผลการวินิจฉัยโรค พร้อมวิธีรักษา หรือ error response
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if image.content_type not in allowed_types:
        return {
            "success": False,
            "error_type": "invalid_file",
            "error": "ประเภทไฟล์ไม่ถูกต้อง",
            "message": "รองรับเฉพาะไฟล์ภาพ JPG, PNG, WEBP เท่านั้น"
        }
    
    # Read image
    try:
        image_bytes = await image.read()
    except Exception as e:
        return {
            "success": False,
            "error_type": "read_error",
            "error": str(e),
            "message": f"ไม่สามารถอ่านไฟล์ได้: {str(e)}"
        }
    
    # Analyze with Vision AI and return result directly
    result = await analyze_leaf_image(image_bytes)
    return result
