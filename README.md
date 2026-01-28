# 🌿 CaneScan - ระบบวินิจฉัยโรคใบอ้อยด้วย AI

<p align="center">
  <img src="https://img.shields.io/badge/Gemini_3-Flash_Preview-blue?style=for-the-badge&logo=google" alt="Gemini 3" />
  <img src="https://img.shields.io/badge/FastAPI-0.109-green?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" alt="License" />
</p>

**Backend API สำหรับวิเคราะห์โรคใบอ้อยจากภาพถ่าย** ใช้ Google Gemini 3 Flash Preview Vision AI ในการวินิจฉัยโรคแบบ Real-time พร้อมคำแนะนำการรักษาและป้องกัน

---

## ✨ Features

- 🔬 **AI Vision Analysis** - วินิจฉัยโรคด้วย Gemini 3 Flash Preview
- 🌿 **รองรับหลายโรค** - Rust, Mosaic, Red Rot, Smut และอื่นๆ
- 💊 **คำแนะนำครบถ้วน** - อาการ, วิธีรักษา, วิธีป้องกัน
- 🌡️ **วิเคราะห์สภาพอากาศ** - ระบุโรคที่เกี่ยวกับสภาพอากาศ
- 🆓 **ฟรี!** - ใช้ Gemini API Free Tier

---

## 📋 โรคที่สามารถวินิจฉัยได้

| ประเภท | โรค |
|--------|-----|
| 🦠 เชื้อไวรัส | Mosaic (ใบด่าง), Yellow Leaf (ใบเหลือง) |
| 🍄 เชื้อรา | Rust (ราสนิม), Red Rot (เน่าแดง), Smut (เขม่าดำ), Brown Spot |
| 🔬 แบคทีเรีย | Leaf Scorch (ใบแห้ง), Ratoon Stunting |
| 🌡️ สภาพอากาศ | Drought Stress, Nutrient Deficiency |
| ✅ สุขภาพดี | Healthy |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/CaneScanDM_BE.git
cd CaneScanDM_BE

# สร้าง Virtual Environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# ติดตั้ง Dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# สร้างไฟล์ .env
cp .env.example .env
```

แก้ไข `.env`:
```env
GEMINI_API_KEY=your_api_key_here
ALLOWED_ORIGINS=http://localhost:5173
```

### 3. Get Gemini API Key (FREE!)

1. ไปที่ [Google AI Studio](https://aistudio.google.com/apikey)
2. คลิก **"Create API Key"**
3. Copy key มาใส่ใน `.env`

### 4. Run Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

🎉 Server พร้อมใช้งานที่ http://localhost:8000

---

## 📡 API Endpoints

### `POST /api/predict`
วิเคราะห์ภาพใบอ้อย

**Request:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "image=@leaf_image.jpg"
```

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "is_sugarcane": true,
    "disease": "Rust",
    "disease_th": "โรคราสนิม",
    "confidence": 0.98,
    "symptoms": ["พบจุดแผลสีน้ำตาลปนส้ม", "แผลพูนพองบนใบ"],
    "analysis": "พบลักษณะของสปอร์เชื้อราบนใบ...",
    "severity": "moderate",
    "cause": "เชื้อรา Puccinia melanocephala",
    "weather_related": true,
    "treatment": ["พ่นสารป้องกันกำจัดเชื้อรา"],
    "prevention": ["เลือกพันธุ์ต้านทานโรค"]
  }
}
```

**Response (Rate Limit):**
```json
{
  "success": false,
  "error_type": "rate_limit",
  "error": "เกิน rate limit",
  "retry_after": 30,
  "message": "กรุณารอ 30 วินาที แล้วลองใหม่"
}
```

### `GET /api/diseases`
ดึงข้อมูลโรคทั้งหมด

### `GET /health`
Health check

---

## 📁 Project Structure

```
CaneScanDM_BE/
├── main.py                 # FastAPI app entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (git ignored)
├── .env.example            # Environment template
├── .gitignore
├── README.md
└── src/
    ├── config.py           # Configuration settings
    ├── routes/
    │   ├── predict.py      # Prediction endpoint
    │   └── diseases.py     # Disease info endpoint
    └── services/
        └── vision_service.py   # Gemini Vision integration
```

---

## ⚠️ Rate Limits (Free Tier)

| Metric | Limit |
|--------|-------|
| Requests/นาที | ~2-5 RPM |
| Requests/วัน | ~20-250 RPD |
| Reset Time | 15:00 เวลาไทย |

> 💡 **Tip:** ถ้าโดน rate limit ให้รอตามเวลาที่ `retry_after` บอก

---

## 🔧 Tech Stack

- **Framework:** FastAPI
- **AI/ML:** Google Gemini 3 Flash Preview
- **Image Processing:** Pillow
- **Language:** Python 3.9+

---

## 🐳 Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t canescan-api .
docker run -p 8000:8000 --env-file .env canescan-api
```

---

## 📝 License

MIT License - ใช้งานได้อิสระ

---

## 👥 Contributors

- Your Name (@yourusername)

---

<p align="center">
  Made with 💚 for Thai Sugarcane Farmers
</p>
