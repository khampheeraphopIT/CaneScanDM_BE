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
# --------------------------------------------------
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={
        "ssl": True,              # ✅ Required for Render/Supabase
        "command_timeout": 60
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
