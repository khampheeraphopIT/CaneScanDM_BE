# CaneScan - Sugarcane Disease Diagnosis

ระบบ AI วินิจฉัยโรคใบอ้อยด้วย Google Gemini 3 Flash ประมวลผลผ่าน Docker เต็มรูปแบบ

## 🚀 วิธีการใช้งานด้วย Docker (แนะนำ)

### 1. เตรียมความพร้อม
- ติดตั้ง [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- สร้างไฟล์ `.env` ในโฟลเดอร์ `DM` (Backend) โดยมีข้อมูลดังนี้:
  ```env
  GEMINI_API_KEY=your_gemini_api_key_here
  ```

### 2. เริ่มต้นระบบ
เปิด Terminal ในโฟลเดอร์ `DM` (Backend) แล้วรันคำสั่ง:
```bash
docker-compose up --build
```

ระบบจะทำการ:
- 🏗️ สร้าง Container สำหรับ **Database** (PostgreSQL 15)
- 🏗️ สร้าง Container สำหรับ **Backend** (FastAPI)
- 🏗️ สร้าง Container สำหรับ **Frontend** (React + Bun)

### 3. เข้าใช้งาน
- **Frontend**: [http://localhost:5173](http://localhost:5173)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Database**: localhost:5432 (User: `postgres`, Pass: `postgres`, DB: `canescan`)

---

## 🛠️ โครงสร้างโปรเจกต์
- `/DM`: Backend (FastAPI, SQLAlchemy, Gemini AI)
- `/DM_web`: Frontend (React, Vite, Bun, TailwindCSS)

## 📦 การติดตั้งแบบ Manual (ไม่ใช้ Docker)

### Backend
1. `cd DM`
2. `python -m venv venv`
3. `venv\Scripts\activate` (Windows)
4. `pip install -r requirements.txt`
5. `uvicorn main:app --reload`

### Frontend
1. `cd DM_web`
2. `bun install`
3. `bun dev`

---

## 📝 หมายเหตุ
- ข้อมูลการวิเคราะห์ทั้งหมดจะถูกบันทึกลงในฐานข้อมูล PostgreSQL ท้องถิ่น (ไม่ใช้ Supabase แล้ว)
- สามารถดูประวัติการวิเคราะห์ได้จากหน้าประวัติในแอปพลิเคชัน
