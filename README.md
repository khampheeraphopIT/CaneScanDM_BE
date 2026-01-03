# Sugarcane Disease Detection (CaneScan DM)

ระบบวิเคราะห์โรคใบอ้อยอัจฉริยะที่ใช้ Deep Learning (ResNet50) ร่วมกับข้อมูลภาพและสภาพอากาศเพื่อความแม่นยำสูงสุด

## 🚀 คุณสมบัติเด่น (Features)

- **High Accuracy Model**: อัปเกรดสถาปัตยกรรมเป็น ResNet50 และความละเอียดภาพ 224x224
- **Hybrid Analysis**: วิเคราะห์ข้อมูลจากทั้ง "รูปถ่าย" และ "ข้อมูลสภาพอากาศ" (อุณหภูมิ, ความชื้น, ปริมาณฝน) แบบ Real-time
- **Multi-Image Support**: รองรับการอัปโหลดและวิเคราะห์หลายรูปภาพพร้อมกันในคำขอเดียว
- **Multi-scale Robustness**: รองรับภาพถ่ายหลายระยะ (ใกล้-ไกล) ด้วยเทคนิค RandomResizedCrop
- **Texture Enhancement**: ใช้ CLAHE ช่วยดึงรายละเอียดของโรคพืชให้ชัดเจนยิ่งขึ้น
- **Automated Logging**: บันทึกประวัติการวิเคราะห์ลงใน CSV อัตโนมัติ

## 🛠️ โครงสร้างโปรเจกต์ (Project Structure)

- `src/model/`: สถาปัตยกรรมโมเดล (`model_arch.py`) และสคริปต์การเทรน (`model.py`)
- `src/routes/`: API Endpoints สำหรับการวิเคราะห์ (`prediction.py`)
- `src/services/`: บริการจัดการข้อมูลและตรรกะการวิเคราะห์
- `uploads/`: โฟลเดอร์เก็บภาพที่ถูกอัปโหลด
- `model/`: โฟลเดอร์เก็บไฟล์โมเดล (`.pth`) และ Scaler (`.joblib`)

---

## 💻 วิธีการติดตั้งและใช้งาน (Installation & Usage)

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. ตั้งค่าสภาพแวดล้อม (Environment Variables)

สร้างไฟล์ `.env` ที่ Root ของโปรเจกต์ และเพิ่มข้อมูล:

```env
OPENWEATHER_API_KEY=your_api_key_here
PORT=8000
```

### 3. การเทรนโมเดลใหม่ (Training)

หากต้องการเริ่มเทรนโมเดลใหม่ด้วยสถาปัตยกรรม ResNet50 ล่าสุด:

```bash
python -m src.model.model
```

_ระบบจะสร้างไฟล์ `model/best_model.pth` และ `model/scaler.joblib` ให้โดยอัตโนมัติ_

### 4. การรันเซิร์ฟเวอร์ (Running the Server)

```bash
python main.py
```

เซิร์ฟเวอร์จะรันที่ `http://localhost:8000`

---

## 📡 API Documentation

### **Analyze Disease**

- **Endpoint**: `POST /predict`
- **Body (form-data)**:
  - `files`: (รองรับหลายไฟล์ภาพ .jpg, .png, .webp)
  - `province`: (ชื่อจังหวัดภาษาไทย เช่น "ลพบุรี", "หนองบัวลำภู")
- **Example Response**:

```json
{
  "timestamp": "2024-12-30T...",
  "province": "ลพบุรี",
  "results": [
    {
      "image": "leaf1.jpg",
      "disease": "Rust",
      "confidence": "98.50%",
      "risk_level": "Moderate",
      "weather": { "temperature": 32.5, "humidity": 65, "rainfall": 0 }
    }
  ]
}
```

---

## 📑 รายละเอียดเพิ่มเติม

อ่านรายละเอียดเทคนิคเชิงลึกเพิ่มเติมได้ที่ `system_documentation.txt`
