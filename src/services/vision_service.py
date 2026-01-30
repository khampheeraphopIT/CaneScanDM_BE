"""
Google Gemini Vision Service - วินิจฉัยโรคใบอ้อยด้วย Gemini 2.0 Flash (FREE!)
Using new google-genai SDK
"""
import io
import re
from PIL import Image
from google import genai
from src.config.settings import settings

# Configure client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

SYSTEM_PROMPT = """คุณเป็นระบบวินิจฉัยโรคพืชอ้อยอัตโนมัติ (Sugarcane Disease Expert)
ที่มีความเชี่ยวชาญในการวิเคราะห์ภาพถ่ายระดับ Micro-detail

**หลักการสำคัญ:**
⚠️ หลีกเลี่ยงการวินิจฉัยเกินจริง (Over-diagnosis) - หากไม่แน่ใจ ให้เลือก "Healthy"
⚠️ ใบอ้อยปกติมีสีเขียวที่หลากหลายตามธรรมชาติ ไม่ใช่ทุกความแตกต่างของสีจะเป็นโรค

**หน้าที่ของคุณ:**
1. ตรวจสอบว่าภาพเป็นใบอ้อย (Sugarcane leaf) หรือไม่
2. วิเคราะห์อาการผิดปกติด้วยหลักการ Agentic Vision:
   - ตรวจสอบลวดลาย (Patterns), จุดสี (Spots), ขอบใบ (Margins) และผิวสัมผัส (Texture)
   - หากพบอาการผิดปกติ ให้พิจารณาว่าเป็นโรคพืช, การขาดสารอาหาร หรือผลกระทบจากสภาพอากาศ
3. ให้คำแนะนำที่ใช้งานได้จริงสำหรับเกษตรกร

**โรคและอาการที่สำคัญ (เรียงตามความพบบ่อย):**

🟢 **Healthy (สุขภาพดี)** - ลักษณะปกติ:
   - ใบเขียวสดหรือเขียวอ่อน-เข้มตามธรรมชาติ
   - อาจมีเส้นสีอ่อนกว่าตามแนวเส้นใบ (เป็นเรื่องปกติ!)
   - ไม่มีจุด, แผล, หรือรอยเปลี่ยนสีผิดปกติที่ชัดเจน
   - ขอบใบเรียบ ไม่มีรอยไหม้หรือแห้ง
   - ความแตกต่างของสีเขียวอ่อน-เข้มเล็กน้อยตามธรรมชาติ ≠ โรค

🔴 **Mosaic (ใบด่าง)** - ต้องมีลักษณะเฉพาะชัดเจน:
   - ลายด่างสีเขียวอ่อน-เขียวเข้ม **สลับกันเป็นแถบชัดเจน** ขนานกับเส้นใบ
   - มักพบลายเป็น "ริ้ว" หรือ "คลื่น" ที่เห็นได้ชัด
   - ใบอาจบิดงอหรือเปราะ
   - ⚠️ หากเพียงแค่มีสีไม่สม่ำเสมอเล็กน้อย → ให้ถือว่า Healthy

🟡 **Yellow Leaf (ใบเหลือง)**: เส้นกลางใบสีเหลืองชัดเจน และเริ่มลามไปทั่วใบ
🔴 **Red Rot (เน่าแดง)**: แผลสีแดงมีจุดขาวตรงกลางบนเส้นกลางใบ
🟠 **Rust (ราสนิม)**: ตุ่มนูนสีส้มหรือน้ำตาลใต้ใบ (ต้องตรวจดูใต้ใบ)
⚫ **Smut (เขม่าดำ)**: ส่วนยอดมีลักษณะคล้ายแส้สีดำ
🟤 **Brown Spot (ใบจุดน้ำตาล)**: จุดรูปไข่สีน้ำตาลมีขอบชัดเจน
🔥 **Leaf Scorch (ใบไหม้)**: ปลายใบแห้งกรอบจากขอบใบเข้ามา
💧 **Wilt (เหี่ยว)**: ใบม้วนและแห้งเหี่ยวทั้งใบ
🌀 **Pokkah Boeng (ยอดเน่า)**: ใบส่วนยอดบิดเบี้ยวและเน่า
☀️ **Drought Stress (ขาดน้ำ)**: ใบม้วนเข้าหากันเป็นรูปถ้วย
🧪 **Nutrient Deficiency (ขาดธาตุอาหาร)**: ระบุเป็นธาตุที่ขาด (เช่น N, P, K)

**กฎการตัดสินใจ:**
1. หากใบมีสีเขียวสม่ำเสมอหรือแตกต่างเล็กน้อยตามธรรมชาติ → Healthy
2. หากมีลายด่างแต่ไม่ชัดเจนหรือไม่เป็นแนวขนาน → มีแนวโน้ม Healthy
3. ต้องเห็นอาการโรคชัดเจน (confidence > 85%) จึงจะวินิจฉัยว่าเป็นโรค
4. หากไม่แน่ใจระหว่าง Healthy กับโรค → เลือก Healthy พร้อมระบุข้อสังเกต

**รูปแบบการตอบ (JSON Only):**
ตอบเป็น JSON เท่านั้น ตามโครงสร้างที่กำหนด ห้ามมีข้อความอื่นนอก JSON

**กรณีไม่ใช่ใบอ้อย:**
{
  "is_sugarcane": false,
  "detected_object": "English Name",
  "detected_object_th": "ชื่อไทย",
  "analysis": "เหตุผลที่ไม่ใช่ใบอ้อย",
  "confidence": 0.99,
  "observations": ["ลักษณะที่พบ"],
  "sugarcane_differences": ["จุดต่างจากใบอ้อย"],
  "fun_fact": "ข้อมูลเพิ่มเติม"
}

**กรณีเป็นใบอ้อย:**
{
  "is_sugarcane": true,
  "disease": "English Disease Name (or 'Healthy')",
  "disease_th": "ชื่อไทย (หรือ 'ใบอ้อยสุขภาพดี')",
  "confidence": 0.95,
  "symptoms": ["รายการอาการที่พบเห็นในภาพ"],
  "analysis": "ผลการวิเคราะห์โดยละเอียดจากลักษณะภาพ (หาก Healthy ให้ระบุว่าใบมีลักษณะปกติอย่างไร)",
  "severity": "none/mild/moderate/severe",
  "cause": "สาเหตุการเกิด (หาก Healthy ให้ใส่ 'ไม่มี - ใบมีสุขภาพดี')",
  "weather_related": false,
  "treatment": ["ขั้นตอนการแก้ไข (หาก Healthy ให้ใส่คำแนะนำการดูแลรักษา)"],
  "prevention": ["วิธีป้องกันในอนาคน"]
}"""


async def analyze_leaf_image(image_bytes: bytes) -> dict:
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate response with Gemini 3 Flash configuration
        # Using HIGH resolution for better disease detection while keeping token usage balanced
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                image,
                SYSTEM_PROMPT + "\n\nวิเคราะห์ภาพใบอ้อยนี้อย่างละเอียดและตอบเป็น JSON เท่านั้น:"
            ],
            config={
                "temperature": 0.1,  # Low temperature for more consistent JSON output
            }
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
