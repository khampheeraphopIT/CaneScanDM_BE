import os
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Get DATABASE_URL
# --------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Local fallback
    DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
elif DATABASE_URL.startswith("postgresql://"):
    # Ensure asyncpg driver
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# --------------------------------------------------
# SQLAlchemy async engine
# --------------------------------------------------
engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,  # ตรวจสอบ connection ก่อนใช้งานทุกครั้ง
    pool_recycle=300,    # รีเซ็ตการเชื่อมต่อทุก 5 นาที
    pool_size=5,         # จำกัดจำนวน connection
    max_overflow=10,     # ขยายได้อีกนิดถ้าจำเป็น
)

# --------------------------------------------------
# Session factory
# --------------------------------------------------
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

# --------------------------------------------------
# Base model
# --------------------------------------------------
class Base(DeclarativeBase):
    pass

# --------------------------------------------------
# Dependency
# --------------------------------------------------
async def get_db():
    async with SessionLocal() as session:
        yield session
