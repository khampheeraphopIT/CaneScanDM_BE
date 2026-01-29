"""
Prediction Route - API endpoint สำหรับวินิจฉัยโรคใบอ้อย
"""
from fastapi import APIRouter, UploadFile, File
from src.services.vision_service import analyze_leaf_image
from src.services.rate_tracker import rate_tracker

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
            "message": "รองรับเฉพาะไฟล์ภาพ JPG, PNG, WEBP เท่านั้น",
            "rate_limit": rate_tracker.to_dict()
        }
    
    # Check local rate limit (pre-flight check)
    rate_info = rate_tracker.to_dict()
    if not rate_info["can_request"]:
        return {
            "success": False,
            "error_type": "rate_limit",
            "error": "กรุณารอสักครู่",
            "retry_after": rate_info["next_available_in"],
            "message": "กรุณารอสักครู่ แล้วลองใหม่อีกครั้ง",
            "rate_limit": rate_info
        }

    # Read image
    try:
        image_bytes = await image.read()
    except Exception as e:
        return {
            "success": False,
            "error_type": "read_error",
            "error": str(e),
            "message": f"ไม่สามารถอ่านไฟล์ได้: {str(e)}",
            "rate_limit": rate_tracker.to_dict()
        }
    
    # Record request locally
    rate_tracker.record_request()
    
    # Analyze with Vision AI
    result = await analyze_leaf_image(image_bytes)
    
    # Handle specific quota errors from API
    if result.get("error_type") == "quota_exceeded":
        rate_tracker.set_daily_quota_exceeded()
    elif result.get("error_type") == "rate_limit" and result.get("retry_after"):
        rate_tracker.set_rate_limited(result["retry_after"])
    
    # Add usage info to every response
    result["rate_limit"] = rate_tracker.to_dict()
    
    return result
