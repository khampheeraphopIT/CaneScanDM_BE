import base64
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from src.services.vision_service import analyze_leaf_image
from src.services.rate_tracker import rate_tracker
from src.models.prediction import PredictionHistory

class PredictionController:
    @staticmethod
    async def predict_disease(image: UploadFile, db: AsyncSession, user_id: int | None = None):
        """
        Business logic for predicting disease from image
        """
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
        content_type = image.content_type or "image/jpeg"
        if content_type not in allowed_types:
            rate_info = await rate_tracker.to_dict()
            return {
                "success": False,
                "error_type": "invalid_file",
                "error": "ประเภทไฟล์ไม่ถูกต้อง",
                "message": "รองรับเฉพาะไฟล์ภาพ JPG, PNG, WEBP เท่านั้น",
                "rate_limit": rate_info
            }
        
        # Check local rate limit (pre-flight check)
        rate_info = await rate_tracker.to_dict()
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
            rate_info = await rate_tracker.to_dict()
            return {
                "success": False,
                "error_type": "read_error",
                "error": str(e),
                "message": f"ไม่สามารถอ่านไฟล์ได้: {str(e)}",
                "rate_limit": rate_info
            }
        
        # Record request locally
        await rate_tracker.record_request()
        
        # Analyze with Vision AI
        result = await analyze_leaf_image(image_bytes)
        
        # Handle specific quota errors from API
        if result.get("error_type") == "quota_exceeded":
            await rate_tracker.set_daily_quota_exceeded()
        elif result.get("error_type") == "rate_limit" and result.get("retry_after"):
            await rate_tracker.set_rate_limited(result["retry_after"])
        
        # Record prediction history if successful
        if result.get("success") and "data" in result:
            try:
                data = result["data"]
                
                # Format recommendation from treatment and prevention lists
                treatment_text = "\n".join(data.get("treatment", []))
                prevention_text = "\n".join(data.get("prevention", []))
                recommendation = f"การรักษา:\n{treatment_text}\n\nการป้องกัน:\n{prevention_text}" if treatment_text or prevention_text else None

                # Encode image as Base64 data URI for storage
                image_data_uri = None
                try:
                    # Reset file position to read again
                    await image.seek(0)
                    image_bytes_for_storage = await image.read()
                    image_base64 = base64.b64encode(image_bytes_for_storage).decode('utf-8')
                    image_data_uri = f"data:{content_type};base64,{image_base64}"
                except Exception as img_err:
                    print(f"Error encoding image: {img_err}")

                new_history = PredictionHistory(
                    user_id=user_id,
                    disease_name=data.get("disease_th") or data.get("disease"),
                    confidence=data.get("confidence"),
                    severity=data.get("severity"),
                    description=data.get("analysis"),
                    recommendation=recommendation,
                    image_url=image_data_uri  # Store Base64 data URI
                )
                db.add(new_history)
                await db.commit()
            except Exception as e:
                print(f"Error saving history: {e}")
                # Don't fail the request if history fails
                await db.rollback()


        # Add usage info to every response
        result["rate_limit"] = await rate_tracker.to_dict()
        
        return result

    @staticmethod
    async def get_history(db: AsyncSession, limit: int = 50):
        """
        Fetch prediction history
        """
        from sqlalchemy import select
        result = await db.execute(select(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(limit))
        history = result.scalars().all()
        return history
