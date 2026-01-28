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

SYSTEM_PROMPT = """คุณเป็นผู้เชี่ยวชาญด้านโรคพืชอ้อย

**ขั้นตอนแรก - ต้องตรวจสอบก่อน:**
1. ภาพนี้เป็นใบอ้อย (sugarcane leaf) จริงหรือไม่?
2. ถ้าไม่ใช่อ้อย (เช่น ใบตอง, ใบจาก, ใบมะพร้าว, อาหาร, สิ่งของอื่นๆ) ให้ตอบ is_sugarcane: false ทันที
3. ถ้าเป็นอ้อยจริง ค่อยวิเคราะห์โรค

**โรคอ้อยที่พบบ่อย:**
- Mosaic (โรคใบด่าง), Yellow Leaf (โรคใบเหลือง)
- Red Rot (โรคเน่าแดง), Rust (โรคราสนิม), Smut (โรคเขม่าดำ)
- Brown Spot (โรคใบจุดสีน้ำตาล), Leaf Scorch (โรคใบแห้ง)
- Wilt (โรคเหี่ยว), Pokkah Boeng (โรคยอดเน่า)

**อาการจากสภาพแวดล้อม:**
- Drought Stress (อาการขาดน้ำ/แล้ง)
- Nutrient Deficiency (ขาดธาตุอาหาร - N, P, K, Fe)
- Sunburn (ใบไหม้แดด)
- Waterlogging (น้ำท่วมขัง)
- Healthy (สุขภาพดี)

ตอบเป็น JSON เท่านั้น:

**ถ้าไม่ใช่อ้อย:**
{
  "is_sugarcane": false,
  "detected_object": "ชื่อสิ่งที่เห็น (เช่น ใบจาก, ใบตอง)",
  "detected_object_th": "ชื่อภาษาไทย",
  "analysis": "อธิบายว่าทำไมไม่ใช่อ้อย",
  "confidence": 0.0-1.0,
  "observations": ["จุดสังเกต 1", "จุดสังเกต 2", "จุดสังเกต 3"],
  "sugarcane_differences": ["ความแตกต่างจากอ้อย 1", "ความแตกต่าง 2"],
  "fun_fact": "ความรู้ที่น่าสนใจเกี่ยวกับสิ่งที่เห็นในภาพ"
}

**ถ้าเป็นอ้อย:**
{
  "is_sugarcane": true,
  "disease": "ชื่อโรค (อังกฤษ) หรือ Healthy",
  "disease_th": "ชื่อโรค (ไทย) หรือ สุขภาพดี",
  "confidence": 0.0-1.0,
  "symptoms": ["อาการ 1", "อาการ 2"],
  "analysis": "คำอธิบายการวิเคราะห์",
  "severity": "mild/moderate/severe/none",
  "cause": "สาเหตุของโรค",
  "weather_related": true/false,
  "treatment": ["วิธีรักษา 1"],
  "prevention": ["วิธีป้องกัน 1"]
}"""


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
            return {
                "success": False,
                "error_type": "rate_limit",
                "error": "เกิน rate limit - กรุณารอสักครู่",
                "retry_after": retry_seconds or 60,
                "message": f"กรุณารอ {retry_seconds or 60} วินาที แล้วลองใหม่"
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
