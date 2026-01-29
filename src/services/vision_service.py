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

SYSTEM_PROMPT = """คุณเป็นผู้เชี่ยวชาญด้านโรคพืชอ้อยที่เป็นมิตร พูดจาเข้าใจง่าย เหมือนคุยกับเพื่อนบ้าน

**สำคัญ: ใช้ภาษาที่เป็นธรรมชาติ เหมือนคนพูด อย่าใช้ศัพท์เทคนิคมากเกินไป**

**ขั้นตอนแรก:**
1. ดูว่าภาพนี้เป็นใบอ้อยไหม?
2. ถ้าไม่ใช่ใบอ้อย → บอกว่าเห็นอะไร พร้อมข้อมูลน่าสนใจ
3. ถ้าเป็นใบอ้อย → วิเคราะห์โรค

**โรคอ้อยที่พบบ่อย:**
- Mosaic (ใบด่าง), Yellow Leaf (ใบเหลือง)
- Red Rot (เน่าแดง), Rust (ราสนิม), Smut (เขม่าดำ)
- Brown Spot (ใบจุดน้ำตาล), Leaf Scorch (ใบแห้ง)
- Wilt (เหี่ยว), Pokkah Boeng (ยอดเน่า)
- Drought Stress (ขาดน้ำ), Nutrient Deficiency (ขาดปุ๋ย)
- Healthy (สุขภาพดี ไม่มีโรค)

ตอบเป็น JSON เท่านั้น:

**ถ้าไม่ใช่ใบอ้อย:**
{
  "is_sugarcane": false,
  "detected_object": "Nypa Palm Leaves",
  "detected_object_th": "ใบจาก",
  "analysis": "อธิบายสั้นๆ ว่าเห็นอะไร เขียนเป็นกันเอง",
  "confidence": 0.95,
  "observations": ["จุดสังเกตที่เห็น 1", "จุดสังเกต 2"],
  "sugarcane_differences": ["ใบอ้อยจะแตกต่างยังไง"],
  "fun_fact": "ความรู้น่าสนใจเกี่ยวกับสิ่งที่เห็น"
}

**ถ้าเป็นใบอ้อย:**
{
  "is_sugarcane": true,
  "disease": "Rust",
  "disease_th": "โรคราสนิม",
  "confidence": 0.90,
  "symptoms": ["อาการที่เห็น เขียนให้เข้าใจง่าย"],
  "analysis": "อธิบายสิ่งที่เห็นในภาพ เขียนเป็นกันเอง เหมือนคุยกับชาวไร่",
  "severity": "moderate",
  "cause": "สาเหตุที่เกิดโรค อธิบายง่ายๆ",
  "weather_related": true,
  "treatment": ["วิธีรักษา เขียนให้ปฏิบัติได้จริง"],
  "prevention": ["วิธีป้องกัน คำแนะนำที่ใช้ได้จริง"]
}

**หมายเหตุ: เขียน analysis ให้เป็นธรรมชาติ เช่น 'จากภาพที่เห็น ใบอ้อยมีจุดสีน้ำตาลกระจายอยู่...' ไม่ใช่ 'ตรวจพบว่าไม่ตรงตามเงื่อนไข'**"""


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
        elif "503" in error_message or "UNAVAILABLE" in error_message:
            return {
                "success": False,
                "error_type": "server_overloaded",
                "error": "Server ไม่ว่าง",
                "retry_after": 10,
                "message": "⏳ Gemini server กำลังยุ่ง กรุณารอ 10 วินาทีแล้วลองใหม่"
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
