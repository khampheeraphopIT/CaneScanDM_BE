"""
Google Gemini Vision Service - วินิจฉัยโรคใบอ้อยด้วย Gemini 2.0 Flash (FREE!)
Using new google-genai SDK
"""
import io
import re
from PIL import Image
from google import genai
from src.config import settings

# Configure client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

SYSTEM_PROMPT = """คุณเป็นระบบวินิจฉัยโรคพืชอ้อยอัตโนมัติ

**รูปแบบการเขียน:** กระชับ ชัดเจน ตรงประเด็น ใช้ภาษาที่เกษตรกรเข้าใจได้

**ขั้นตอน:**
1. ตรวจสอบว่าภาพเป็นใบอ้อยหรือไม่
2. ถ้าไม่ใช่ใบอ้อย → ระบุสิ่งที่ตรวจพบ
3. ถ้าเป็นใบอ้อย → วิเคราะห์โรคหรืออาการผิดปกติ

**โรคที่รองรับ:**
- Mosaic (ใบด่าง), Yellow Leaf (ใบเหลือง)
- Red Rot (เน่าแดง), Rust (ราสนิม), Smut (เขม่าดำ)
- Brown Spot (ใบจุดน้ำตาล), Leaf Scorch (ใบแห้ง)
- Wilt (เหี่ยว), Pokkah Boeng (ยอดเน่า)
- Drought Stress (ขาดน้ำ), Nutrient Deficiency (ขาดธาตุอาหาร)
- Healthy (สุขภาพดี)

ตอบเป็น JSON เท่านั้น:

**กรณีไม่ใช่ใบอ้อย:**
{
  "is_sugarcane": false,
  "detected_object": "ชื่อภาษาอังกฤษ",
  "detected_object_th": "ชื่อภาษาไทย",
  "analysis": "อธิบายสิ่งที่ตรวจพบ",
  "confidence": 0.95,
  "observations": ["ลักษณะที่สังเกตได้"],
  "sugarcane_differences": ["ความแตกต่างจากใบอ้อย"],
  "fun_fact": "ข้อมูลเพิ่มเติมที่น่าสนใจ"
}

**กรณีเป็นใบอ้อย:**
{
  "is_sugarcane": true,
  "disease": "ชื่อโรคภาษาอังกฤษ",
  "disease_th": "ชื่อโรคภาษาไทย",
  "confidence": 0.90,
  "symptoms": ["อาการที่ตรวจพบ"],
  "analysis": "สรุปผลการวิเคราะห์จากลักษณะที่พบในภาพ",
  "severity": "mild/moderate/severe/none",
  "cause": "สาเหตุของโรค",
  "weather_related": true,
  "treatment": ["คำแนะนำการรักษา"],
  "prevention": ["วิธีป้องกัน"]
}

**หมายเหตุ:** เขียนให้กระชับ เน้นข้อมูลที่เป็นประโยชน์ต่อการตัดสินใจของเกษตรกร"""


async def analyze_leaf_image(image_bytes: bytes) -> dict:
    """
    วิเคราะห์ภาพใบอ้อยด้วย Gemini 2.0 Flash (FREE!)
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate response with PIL Image
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                SYSTEM_PROMPT + "\n\nวิเคราะห์ภาพใบอ้อยนี้ และตอบเป็น JSON เท่านั้น:",
                image
            ]
        )
        
        result_text = response.text
        
        # Parse JSON from response
        import json
        # Clean up response if wrapped in markdown
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result = json.loads(result_text.strip())
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        error_message = str(e)
        print(f"[ERROR] Gemini API: {error_message}")
        
        # Extract retry time if available
        retry_seconds = None
        retry_match = re.search(r'retry in (\d+)', error_message)
        if retry_match:
            retry_seconds = int(retry_match.group(1))
        
        # Check for rate limit / quota errors
        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            is_daily_quota = "RESOURCE_EXHAUSTED" in error_message or "quota" in error_message.lower()
            return {
                "success": False,
                "error_type": "quota_exceeded" if is_daily_quota else "rate_limit",
                "error": "โควต้าประจำวันหมด" if is_daily_quota else "เกิน rate limit - กรุณารอสักครู่",
                "retry_after": retry_seconds or (3600 if is_daily_quota else 60),
                "message": "โควต้าวันนี้หมดแล้ว (20/20) กรุณารอวันใหม่" if is_daily_quota else "เกินขีดจำกัดการใช้งานชั่วคราว กรุณารอสักครู่"
            }
        elif "503" in error_message or "UNAVAILABLE" in error_message:
            return {
                "success": False,
                "error_type": "server_overloaded",
                "error": "Server ไม่ว่าง",
                "retry_after": 10,
                "message": "⏳ เซิร์ฟเวอร์กำลังยุ่ง กรุณารอสักครู่แล้วลองใหม่"
            }
        elif "quota" in error_message.lower():
            return {
                "success": False,
                "error_type": "quota_exceeded",
                "error": "Quota หมด",
                "message": "Quota ประจำวันหมด กรุณารอวันใหม่หรือเปลี่ยน API key"
            }
        
        return {
            "success": False,
            "error_type": "unknown",
            "error": error_message,
            "message": f"เกิดข้อผิดพลาด: {error_message}"
        }
