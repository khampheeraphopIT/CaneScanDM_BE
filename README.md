# 🌿 CaneScan Backend API

**ระบบวินิจฉัยโรคใบอ้อยด้วย AI** - Backend API ใช้ Google Gemini Vision สำหรับวิเคราะห์โรคจากภาพใบอ้อย

## ✨ Features

- 🔬 วินิจฉัยโรคใบอ้อยจากภาพด้วย AI (Gemini 2.0 Flash)
- 🌡️ รองรับโรคจากสภาพอากาศและโรคทั่วไป
- 💊 แนะนำวิธีรักษาและป้องกัน
- 🆓 ใช้ Gemini API ฟรี (15 requests/นาที, 1500/วัน)

## 📋 โรคที่สามารถวินิจฉัยได้

| ประเภท | โรค |
|--------|-----|
| เชื้อไวรัส | Mosaic, Yellow Leaf Syndrome, Streak Mosaic |
| เชื้อรา | Red Rot, Rust, Smut, Brown Spot, Wilt |
| แบคทีเรีย | Leaf Scorch, Ratoon Stunting |
| สภาพอากาศ | Drought Stress, Nutrient Deficiency |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Google Gemini API Key (ฟรี!)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/CaneScanDM_BE.git
cd CaneScanDM_BE

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. สร้างไฟล์ `.env` จาก template:
```bash
cp .env.example .env
```

2. เพิ่ม API Key ใน `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

### Get Gemini API Key (FREE!)

1. ไปที่ https://makersuite.google.com/app/apikey
2. คลิก "Create API Key"
3. Copy key มาใส่ใน `.env`

### Run Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server จะรันที่ http://localhost:8000

## 📡 API Endpoints

### POST /api/predict
วิเคราะห์ภาพใบอ้อย

**Request:**
```
Content-Type: multipart/form-data
Body: image (file)
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_sugarcane": true,
    "disease": "Red Rot",
    "disease_th": "โรคเน่าแดง",
    "confidence": 0.92,
    "symptoms": ["ใบมีจุดสีแดง", "ลำต้นเน่า"],
    "severity": "moderate",
    "treatment": ["ตัดส่วนที่เป็นโรคทิ้ง"],
    "prevention": ["ใช้พันธุ์ต้านทานโรค"]
  }
}
```

### GET /api/diseases
ดึงข้อมูลโรคทั้งหมด

### GET /health
Health check

## 📁 Project Structure

```
├── main.py              # FastAPI app entry
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
└── src/
    ├── config.py        # Configuration
    ├── routes/
    │   ├── predict.py   # Prediction endpoint
    │   └── diseases.py  # Diseases info endpoint
    └── services/
        └── vision_service.py  # Gemini Vision integration
```

## 🔧 Tech Stack

- **Framework:** FastAPI
- **AI/ML:** Google Gemini 2.0 Flash
- **Language:** Python 3.9+

## 📝 License

MIT License

## 👥 Contributors

- Your Name (@yourusername)

---

Made with 💚 for Thai Sugarcane Farmers
