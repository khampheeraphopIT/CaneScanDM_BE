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

if DATABASE_URL:
    # Ensure asyncpg driver
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1
        )

    # 🔥 asyncpg does NOT support sslmode → remove it
    parsed = urlparse(DATABASE_URL)
    query = dict(parse_qsl(parsed.query))
    query.pop("sslmode", None)

    DATABASE_URL = urlunparse(
        parsed._replace(query=urlencode(query))
    )
else:
    # Local fallback
    DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"

# --------------------------------------------------
# SQLAlchemy async engine
# Build engine with extra protection for production
engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,  # ตรวจสอบ connection ก่อนใช้งานทุกครั้ง
    pool_recycle=300,    # รีเซ็ตการเชื่อมต่อทุก 5 นาที
    pool_size=5,         # จำกัดจำนวน connection (ฟรีแพลนจำกัดจำนวน)
    max_overflow=10,     # ขยายได้อีกนิดหน่อยถ้าจำเป็น
    connect_args={
        "ssl": True if "localhost" not in DATABASE_URL else False,
        "command_timeout": 60,
        "server_settings": {
            "tcp_user_timeout": "30000",
        }
    }
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
