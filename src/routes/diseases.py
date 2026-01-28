"""
Diseases Route - ข้อมูลโรคอ้อยและวิธีรักษา
"""
from fastapi import APIRouter

router = APIRouter()

DISEASES = {
    "healthy": {
        "name": "Healthy",
        "name_th": "สุขภาพดี",
        "description": "ใบอ้อยมีสุขภาพดี ไม่พบอาการของโรค",
        "symptoms": [],
        "treatment": ["ดูแลรักษาตามปกติ", "ให้น้ำและปุ๋ยอย่างเหมาะสม"],
        "prevention": ["ตรวจสอบแปลงอย่างสม่ำเสมอ", "ใช้พันธุ์ที่ต้านทานโรค"],
        "severity_color": "#4CAF50"
    },
    "mosaic": {
        "name": "Mosaic",
        "name_th": "โรคใบด่าง",
        "description": "เกิดจากเชื้อไวรัส Sugarcane Mosaic Virus (SCMV) ทำให้ใบมีลายด่างสีเขียวอ่อนสลับเขียวเข้ม",
        "symptoms": [
            "ใบมีลายด่างสีเขียวอ่อนสลับเขียวเข้ม",
            "ใบอาจบิดงอหรือเปลี่ยนรูป",
            "การเจริญเติบโตชะงัก"
        ],
        "treatment": [
            "ขุดทำลายต้นที่เป็นโรค",
            "ใช้ท่อนพันธุ์ที่ปลอดโรค",
            "ควบคุมเพลี้ยอ่อนที่เป็นพาหะ"
        ],
        "prevention": [
            "ใช้พันธุ์ต้านทาน",
            "กำจัดวัชพืชที่เป็นแหล่งอาศัยของเพลี้ย",
            "ฉีดพ่นสารกำจัดเพลี้ยอ่อน"
        ],
        "severity_color": "#FF9800"
    },
    "red_rot": {
        "name": "Red Rot",
        "name_th": "โรคเน่าแดง",
        "description": "เกิดจากเชื้อรา Colletotrichum falcatum ทำให้ลำต้นเน่าเป็นสีแดง",
        "symptoms": [
            "ใบแห้งเหี่ยวจากปลายใบ",
            "ลำต้นภายในเป็นสีแดง มีกลิ่นเหม็น",
            "ต้นตายทั้งกอ"
        ],
        "treatment": [
            "ขุดทำลายต้นที่เป็นโรคทันที",
            "ไม่ใช้ท่อนพันธุ์จากแปลงที่เป็นโรค",
            "แช่ท่อนพันธุ์ในน้ำร้อน 52°C นาน 30 นาที"
        ],
        "prevention": [
            "ใช้พันธุ์ต้านทาน",
            "หลีกเลี่ยงการปลูกซ้ำในพื้นที่เดิม",
            "ระบายน้ำไม่ให้ขังแปลง"
        ],
        "severity_color": "#F44336"
    },
    "rust": {
        "name": "Rust",
        "name_th": "โรคราสนิม",
        "description": "เกิดจากเชื้อรา Puccinia melanocephala ทำให้ใบมีจุดสีน้ำตาลแดงคล้ายสนิม",
        "symptoms": [
            "มีจุดเล็กๆ สีเหลืองน้ำตาลบนใบ",
            "จุดขยายใหญ่เป็นแถบสีน้ำตาลแดง",
            "ใบแห้งตายหากระบาดรุนแรง"
        ],
        "treatment": [
            "ฉีดพ่นสารป้องกันกำจัดเชื้อรา เช่น Mancozeb",
            "ตัดใบที่เป็นโรคทำลาย",
            "ลดความชื้นในแปลง"
        ],
        "prevention": [
            "ใช้พันธุ์ต้านทาน",
            "ปลูกระยะห่างที่เหมาะสม",
            "หลีกเลี่ยงการให้น้ำบนใบ"
        ],
        "severity_color": "#FF5722"
    },
    "yellow": {
        "name": "Yellow Leaf",
        "name_th": "โรคใบเหลือง",
        "description": "เกิดจากเชื้อ Sugarcane Yellow Leaf Virus (SCYLV) ทำให้ใบเหลืองโดยเฉพาะเส้นกลางใบ",
        "symptoms": [
            "ใบเหลืองโดยเฉพาะที่เส้นกลางใบ",
            "ปลายใบแห้ง",
            "ต้นแคระแกร็น น้ำหนักลดลง"
        ],
        "treatment": [
            "ขุดทำลายต้นที่เป็นโรค",
            "ใช้ท่อนพันธุ์ที่ปลอดโรค",
            "ควบคุมเพลี้ยอ่อนที่เป็นพาหะ"
        ],
        "prevention": [
            "ใช้พันธุ์ต้านทาน",
            "ทำความสะอาดเครื่องมือตัด",
            "กำจัดเพลี้ยอ่อนในแปลง"
        ],
        "severity_color": "#FFC107"
    },
    "not_sugarcane": {
        "name": "Not Sugarcane",
        "name_th": "ไม่ใช่ใบอ้อย",
        "description": "ภาพที่อัปโหลดไม่ใช่ใบอ้อย กรุณาอัปโหลดภาพใบอ้อย",
        "symptoms": [],
        "treatment": ["กรุณาถ่ายภาพใบอ้อยและลองอีกครั้ง"],
        "prevention": [],
        "severity_color": "#9E9E9E"
    }
}


@router.get("/diseases")
async def get_diseases():
    """
    ดึงข้อมูลโรคอ้อยทั้งหมด
    """
    return {
        "success": True,
        "diseases": DISEASES
    }


@router.get("/diseases/{disease_key}")
async def get_disease(disease_key: str):
    """
    ดึงข้อมูลโรคตาม key
    """
    disease_key = disease_key.lower().replace(" ", "_")
    
    if disease_key not in DISEASES:
        return {
            "success": False,
            "error": "ไม่พบข้อมูลโรค"
        }
    
    return {
        "success": True,
        "disease": DISEASES[disease_key]
    }
